"""
Music Recommender Engine — Hybrid (Content-Based + Collaborative Filtering)

Uses scikit-learn for:
- KNN content-based similarity on audio features
- NearestNeighbors collaborative filtering on user-item ratings
- Cosine similarity for hybrid score blending

Includes confidence scoring, logging, and full explainability.
"""

import csv
import logging
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

handlers = [logging.StreamHandler()]
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    handlers.append(logging.FileHandler(os.path.join(LOG_DIR, "recommender.log")))
except OSError:
    pass  # If logs folder can't be created (e.g. OneDrive), just log to console

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("MusicRecommender")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its audio features."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


@dataclass
class Recommendation:
    """A single recommendation with metadata."""
    song: Song
    hybrid_score: float
    content_score: float
    collaborative_score: float
    confidence: float
    explanation: str

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Song]:
    """Load songs from CSV into Song objects."""
    songs = []
    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    mood=row["mood"],
                    energy=float(row["energy"]),
                    tempo_bpm=float(row["tempo_bpm"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                ))
        logger.info("Loaded %d songs from %s", len(songs), csv_path)
    except FileNotFoundError:
        logger.error("Song file not found: %s", csv_path)
        raise
    except Exception as e:
        logger.error("Error loading songs: %s", e)
        raise
    return songs


def load_ratings(csv_path: str) -> List[Dict]:
    """Load user ratings from CSV."""
    ratings = []
    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratings.append({
                    "user_id": int(row["user_id"]),
                    "song_id": int(row["song_id"]),
                    "rating": float(row["rating"]),
                })
        logger.info("Loaded %d ratings from %s", len(ratings), csv_path)
    except FileNotFoundError:
        logger.warning("Ratings file not found: %s — collaborative filtering disabled", csv_path)
    except Exception as e:
        logger.error("Error loading ratings: %s", e)
    return ratings

# ---------------------------------------------------------------------------
# Content-Based Engine (scikit-learn KNN)
# ---------------------------------------------------------------------------

class ContentEngine:
    """Content-based recommender using KNN on audio features."""

    FEATURE_KEYS = ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]

    def __init__(self, songs: List[Song]):
        self.songs = songs
        self.scaler = StandardScaler()
        self._feature_matrix = None
        self._knn = None
        self._fit()

    def _fit(self):
        raw = np.array([
            [getattr(s, k) for k in self.FEATURE_KEYS] for s in self.songs
        ])
        self._feature_matrix = self.scaler.fit_transform(raw)
        self._knn = NearestNeighbors(n_neighbors=min(20, len(self.songs)), metric="cosine")
        self._knn.fit(self._feature_matrix)
        logger.info("ContentEngine fitted on %d songs with %d features",
                     len(self.songs), len(self.FEATURE_KEYS))

    def query(self, user: UserProfile, k: int = 10) -> List[Tuple[Song, float]]:
        """Find songs closest to user preference vector."""
        # Build a synthetic preference vector
        pref_raw = np.array([[
            user.target_energy,
            120.0,  # neutral tempo
            0.75 if user.favorite_mood in ("happy",) else 0.45,
            0.70,
            0.80 if user.likes_acoustic else 0.20,
        ]])
        pref_scaled = self.scaler.transform(pref_raw)

        distances, indices = self._knn.kneighbors(pref_scaled, n_neighbors=min(k * 2, len(self.songs)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            song = self.songs[idx]
            # Boost for genre / mood match
            genre_bonus = 0.0
            if song.genre == user.favorite_genre:
                genre_bonus = 0.25
            elif user.favorite_genre in song.genre or song.genre in user.favorite_genre:
                genre_bonus = 0.12
            mood_bonus = 0.20 if song.mood == user.favorite_mood else 0.0

            # Convert cosine distance to similarity (0-1), add bonuses
            similarity = max(0.0, 1.0 - dist) + genre_bonus + mood_bonus
            results.append((song, round(similarity, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def song_similarity(self, song_a: Song, song_b: Song) -> float:
        """Cosine similarity between two songs."""
        idx_a = next(i for i, s in enumerate(self.songs) if s.id == song_a.id)
        idx_b = next(i for i, s in enumerate(self.songs) if s.id == song_b.id)
        sim = cosine_similarity(
            self._feature_matrix[idx_a].reshape(1, -1),
            self._feature_matrix[idx_b].reshape(1, -1),
        )
        return float(sim[0][0])

# ---------------------------------------------------------------------------
# Collaborative Filtering Engine (scikit-learn NearestNeighbors)
# ---------------------------------------------------------------------------

class CollaborativeEngine:
    """Item-based collaborative filtering using NearestNeighbors on a user-item matrix."""

    def __init__(self, songs: List[Song], ratings: List[Dict]):
        self.songs = songs
        self.ratings = ratings
        self._song_id_to_idx = {s.id: i for i, s in enumerate(songs)}
        self._matrix = None
        self._knn = None
        self._fitted = False
        if ratings:
            self._fit()

    def _fit(self):
        user_ids = sorted(set(r["user_id"] for r in self.ratings))
        user_idx_map = {uid: i for i, uid in enumerate(user_ids)}

        n_users = len(user_ids)
        n_songs = len(self.songs)
        self._matrix = np.zeros((n_songs, n_users))

        for r in self.ratings:
            song_idx = self._song_id_to_idx.get(r["song_id"])
            user_idx = user_idx_map.get(r["user_id"])
            if song_idx is not None and user_idx is not None:
                self._matrix[song_idx, user_idx] = r["rating"]

        self._knn = NearestNeighbors(n_neighbors=min(10, n_songs), metric="cosine")
        self._knn.fit(self._matrix)
        self._fitted = True
        logger.info("CollaborativeEngine fitted: %d songs × %d users", n_songs, n_users)

    def query(self, liked_song_ids: List[int], k: int = 10) -> List[Tuple[Song, float]]:
        """Given songs a user liked, find collaboratively similar songs."""
        if not self._fitted or not liked_song_ids:
            return []

        # Average the rating vectors of liked songs
        vectors = []
        for sid in liked_song_ids:
            idx = self._song_id_to_idx.get(sid)
            if idx is not None:
                vectors.append(self._matrix[idx])
        if not vectors:
            return []

        avg_vector = np.mean(vectors, axis=0).reshape(1, -1)
        distances, indices = self._knn.kneighbors(avg_vector, n_neighbors=min(k * 2, len(self.songs)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            song = self.songs[idx]
            if song.id not in liked_song_ids:
                similarity = max(0.0, 1.0 - dist)
                results.append((song, round(similarity, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

# ---------------------------------------------------------------------------
# Hybrid Recommender
# ---------------------------------------------------------------------------

class HybridRecommender:
    """
    Combines content-based and collaborative filtering with configurable weights.
    Provides confidence scores and explanations for every recommendation.
    """

    def __init__(
        self,
        songs: List[Song],
        ratings: Optional[List[Dict]] = None,
        content_weight: float = 0.6,
        collab_weight: float = 0.4,
    ):
        if not songs:
            raise ValueError("Song catalog cannot be empty")
        if not (0 <= content_weight <= 1) or not (0 <= collab_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.songs = songs
        self.ratings = ratings or []
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_engine = ContentEngine(songs)
        self.collab_engine = CollaborativeEngine(songs, self.ratings)

        self._song_map = {s.id: s for s in songs}
        logger.info(
            "HybridRecommender initialized: %d songs, %d ratings, "
            "weights=(content=%.2f, collab=%.2f)",
            len(songs), len(self.ratings), content_weight, collab_weight,
        )

    def recommend(
        self,
        user: UserProfile,
        liked_song_ids: Optional[List[int]] = None,
        k: int = 5,
    ) -> List[Recommendation]:
        """
        Generate hybrid recommendations with confidence scores.
        """
        liked_song_ids = liked_song_ids or []
        logger.info(
            "Generating recommendations: genre=%s, mood=%s, energy=%.2f, "
            "liked_songs=%s, k=%d",
            user.favorite_genre, user.favorite_mood, user.target_energy,
            liked_song_ids, k,
        )

        # --- Content-based scores ---
        content_results = self.content_engine.query(user, k=len(self.songs))
        content_scores = {song.id: score for song, score in content_results}

        # --- Collaborative scores ---
        collab_scores = {}
        if liked_song_ids and self.collab_engine._fitted:
            collab_results = self.collab_engine.query(liked_song_ids, k=len(self.songs))
            collab_scores = {song.id: score for song, score in collab_results}

        # --- Blend ---
        has_collab = bool(collab_scores)
        cw = self.content_weight if has_collab else 1.0
        cfw = self.collab_weight if has_collab else 0.0

        hybrid = {}
        for song in self.songs:
            cs = content_scores.get(song.id, 0.0)
            cf = collab_scores.get(song.id, 0.0)
            hybrid[song.id] = (cs, cf, cw * cs + cfw * cf)

        # Sort by hybrid score
        ranked_ids = sorted(hybrid, key=lambda sid: hybrid[sid][2], reverse=True)

        # --- Build recommendations ---
        recommendations = []
        # Normalise hybrid scores for confidence
        all_hybrid_scores = [hybrid[sid][2] for sid in ranked_ids]
        max_score = max(all_hybrid_scores) if all_hybrid_scores else 1.0
        min_score = min(all_hybrid_scores) if all_hybrid_scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        for sid in ranked_ids[:k]:
            if sid in [s_id for s_id in liked_song_ids]:
                continue
            song = self._song_map[sid]
            cs, cf, hs = hybrid[sid]

            # Confidence: normalised hybrid score (0-1)
            confidence = round((hs - min_score) / score_range, 3)
            # Clamp
            confidence = max(0.0, min(1.0, confidence))

            explanation = self._explain(user, song, cs, cf, has_collab)

            recommendations.append(Recommendation(
                song=song,
                hybrid_score=round(hs, 4),
                content_score=round(cs, 4),
                collaborative_score=round(cf, 4),
                confidence=confidence,
                explanation=explanation,
            ))

        logger.info("Returned %d recommendations", len(recommendations))
        return recommendations[:k]

    def _explain(self, user: UserProfile, song: Song, cs: float, cf: float, has_collab: bool) -> str:
        reasons = []

        if song.genre == user.favorite_genre:
            reasons.append("matches your favorite genre exactly")
        elif user.favorite_genre in song.genre or song.genre in user.favorite_genre:
            reasons.append("is close to your favorite genre")

        if song.mood == user.favorite_mood:
            reasons.append("matches your preferred mood")

        energy_diff = abs(song.energy - user.target_energy)
        if energy_diff <= 0.10:
            reasons.append("energy level is a very close match")
        elif energy_diff <= 0.25:
            reasons.append("energy level is fairly close")

        if user.likes_acoustic and song.acousticness >= 0.60:
            reasons.append("fits your acoustic preference")
        elif not user.likes_acoustic and song.acousticness <= 0.40:
            reasons.append("fits your non-acoustic preference")

        if song.danceability >= 0.75:
            reasons.append("highly danceable")
        if song.valence >= 0.75:
            reasons.append("upbeat feel")

        if has_collab and cf > 0.3:
            reasons.append("liked by users with similar taste")

        if not reasons:
            return "Recommended based on overall similarity to your preferences."

        return "Recommended because it " + ", ".join(reasons) + "."

    def get_available_genres(self) -> List[str]:
        return sorted(set(s.genre for s in self.songs))

    def get_available_moods(self) -> List[str]:
        return sorted(set(s.mood for s in self.songs))

    def get_song_by_id(self, song_id: int) -> Optional[Song]:
        return self._song_map.get(song_id)

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_precision_at_k(recommender: HybridRecommender, user: UserProfile,
                            relevant_song_ids: List[int], k: int = 5) -> float:
    """Precision@K: fraction of top-K recommendations that are in the relevant set."""
    recs = recommender.recommend(user, k=k)
    hits = sum(1 for r in recs if r.song.id in relevant_song_ids)
    precision = hits / k if k > 0 else 0.0
    logger.info("Precision@%d = %.2f (%d/%d hits)", k, precision, hits, k)
    return precision


def evaluate_genre_diversity(recommendations: List[Recommendation]) -> float:
    """Genre diversity: unique genres / total recommendations."""
    if not recommendations:
        return 0.0
    genres = set(r.song.genre for r in recommendations)
    diversity = len(genres) / len(recommendations)
    logger.info("Genre diversity = %.2f (%d unique genres in %d recs)",
                diversity, len(genres), len(recommendations))
    return diversity


def evaluate_avg_confidence(recommendations: List[Recommendation]) -> float:
    """Average confidence score across recommendations."""
    if not recommendations:
        return 0.0
    avg = sum(r.confidence for r in recommendations) / len(recommendations)
    logger.info("Average confidence = %.3f", avg)
    return avg


def evaluate_coverage(recommender: HybridRecommender, users: List[UserProfile],
                      k: int = 5) -> float:
    """Catalog coverage: fraction of songs that appear in at least one top-K list."""
    recommended_ids = set()
    for user in users:
        recs = recommender.recommend(user, k=k)
        for r in recs:
            recommended_ids.add(r.song.id)
    coverage = len(recommended_ids) / len(recommender.songs) if recommender.songs else 0.0
    logger.info("Coverage = %.2f (%d/%d songs recommended)",
                coverage, len(recommended_ids), len(recommender.songs))
    return coverage
