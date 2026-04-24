"""
Comprehensive test suite for the Hybrid Music Recommender.

Covers:
- Data loading
- Content-based engine
- Collaborative filtering engine
- Hybrid blending
- Confidence scoring
- Evaluation metrics
- Edge cases and error handling
"""

import os
import sys
try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recommender import (
    Song, UserProfile, Recommendation,
    ContentEngine, CollaborativeEngine, HybridRecommender,
    load_songs, load_ratings,
    evaluate_precision_at_k, evaluate_genre_diversity, evaluate_avg_confidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SONGS = [
    Song(1, "Pop Hit", "Artist A", "pop", "happy", 0.82, 120, 0.85, 0.80, 0.15),
    Song(2, "Chill Lofi", "Artist B", "lofi", "chill", 0.40, 75, 0.55, 0.60, 0.80),
    Song(3, "Hard Rock", "Artist C", "rock", "intense", 0.92, 155, 0.45, 0.65, 0.08),
    Song(4, "Jazz Night", "Artist D", "jazz", "relaxed", 0.35, 85, 0.70, 0.50, 0.90),
    Song(5, "EDM Banger", "Artist E", "edm", "intense", 0.95, 140, 0.70, 0.92, 0.03),
    Song(6, "Ambient Drift", "Artist F", "ambient", "chill", 0.25, 55, 0.50, 0.30, 0.95),
    Song(7, "Indie Pop", "Artist G", "indie pop", "happy", 0.72, 118, 0.78, 0.75, 0.40),
    Song(8, "Folk Song", "Artist H", "folk", "relaxed", 0.33, 90, 0.72, 0.48, 0.88),
]

SAMPLE_RATINGS = [
    {"user_id": 1, "song_id": 1, "rating": 5},
    {"user_id": 1, "song_id": 5, "rating": 4},
    {"user_id": 1, "song_id": 7, "rating": 5},
    {"user_id": 2, "song_id": 2, "rating": 5},
    {"user_id": 2, "song_id": 4, "rating": 4},
    {"user_id": 2, "song_id": 6, "rating": 5},
    {"user_id": 3, "song_id": 3, "rating": 5},
    {"user_id": 3, "song_id": 5, "rating": 4},
    {"user_id": 3, "song_id": 1, "rating": 3},
]

POP_USER = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)

CHILL_USER = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                         target_energy=0.4, likes_acoustic=True)

ROCK_USER = UserProfile(favorite_genre="rock", favorite_mood="intense",
                        target_energy=0.9, likes_acoustic=False)


# ---------------------------------------------------------------------------
# 1. Data loading tests
# ---------------------------------------------------------------------------

def test_load_songs_from_csv():
    """Songs CSV loads correctly with all fields."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    songs = load_songs(os.path.join(data_dir, "songs.csv"))
    assert len(songs) == 100
    assert all(isinstance(s, Song) for s in songs)
    assert songs[0].title == "Sunrise City"


def test_load_songs_missing_file():
    """Missing file raises FileNotFoundError."""
    try:
        load_songs("nonexistent.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_ratings_from_csv():
    """Ratings CSV loads correctly."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    ratings = load_ratings(os.path.join(data_dir, "user_ratings.csv"))
    assert len(ratings) > 0
    assert all("user_id" in r and "song_id" in r and "rating" in r for r in ratings)


# ---------------------------------------------------------------------------
# 2. Content-based engine tests
# ---------------------------------------------------------------------------

def test_content_engine_returns_correct_count():
    """ContentEngine returns the requested number of results."""
    engine = ContentEngine(SAMPLE_SONGS)
    results = engine.query(POP_USER, k=3)
    assert len(results) == 3
    assert all(isinstance(s, Song) for s, _ in results)


def test_content_engine_pop_user_prefers_pop():
    """A pop/happy user should get pop-like songs ranked higher."""
    engine = ContentEngine(SAMPLE_SONGS)
    results = engine.query(POP_USER, k=3)
    top_genres = [s.genre for s, _ in results]
    # The top result should be pop or indie pop (closest)
    assert top_genres[0] in ("pop", "indie pop")


def test_content_engine_similarity_symmetric():
    """Song similarity should be symmetric: sim(A,B) == sim(B,A)."""
    engine = ContentEngine(SAMPLE_SONGS)
    sim_ab = engine.song_similarity(SAMPLE_SONGS[0], SAMPLE_SONGS[1])
    sim_ba = engine.song_similarity(SAMPLE_SONGS[1], SAMPLE_SONGS[0])
    assert abs(sim_ab - sim_ba) < 1e-6


# ---------------------------------------------------------------------------
# 3. Collaborative filtering tests
# ---------------------------------------------------------------------------

def test_collaborative_engine_returns_results():
    """CollaborativeEngine returns recommendations given liked songs."""
    engine = CollaborativeEngine(SAMPLE_SONGS, SAMPLE_RATINGS)
    results = engine.query(liked_song_ids=[1, 7], k=3)
    assert len(results) > 0
    # Should not include the liked songs themselves
    returned_ids = [s.id for s, _ in results]
    assert 1 not in returned_ids
    assert 7 not in returned_ids


def test_collaborative_engine_no_ratings():
    """CollaborativeEngine handles empty ratings gracefully."""
    engine = CollaborativeEngine(SAMPLE_SONGS, [])
    results = engine.query(liked_song_ids=[1], k=3)
    assert results == []


# ---------------------------------------------------------------------------
# 4. Hybrid recommender tests
# ---------------------------------------------------------------------------

def test_hybrid_returns_recommendations():
    """Hybrid recommender returns Recommendation objects with all fields."""
    rec = HybridRecommender(SAMPLE_SONGS, SAMPLE_RATINGS)
    recs = rec.recommend(POP_USER, liked_song_ids=[1], k=5)
    assert len(recs) > 0
    assert all(isinstance(r, Recommendation) for r in recs)
    for r in recs:
        assert 0 <= r.confidence <= 1
        assert r.explanation != ""
        assert r.hybrid_score >= 0


def test_hybrid_empty_catalog_raises():
    """Empty song list raises ValueError."""
    try:
        HybridRecommender([], [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_hybrid_scores_sorted_descending():
    """Recommendations are sorted by hybrid score, highest first."""
    rec = HybridRecommender(SAMPLE_SONGS, SAMPLE_RATINGS)
    recs = rec.recommend(ROCK_USER, k=5)
    scores = [r.hybrid_score for r in recs]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 5. Confidence scoring tests
# ---------------------------------------------------------------------------

def test_confidence_between_zero_and_one():
    """All confidence scores are in [0, 1]."""
    rec = HybridRecommender(SAMPLE_SONGS, SAMPLE_RATINGS)
    recs = rec.recommend(CHILL_USER, k=5)
    for r in recs:
        assert 0.0 <= r.confidence <= 1.0


def test_top_recommendation_has_highest_confidence():
    """The first recommendation should have the highest confidence."""
    rec = HybridRecommender(SAMPLE_SONGS, SAMPLE_RATINGS)
    recs = rec.recommend(POP_USER, k=5)
    if len(recs) >= 2:
        assert recs[0].confidence >= recs[-1].confidence


# ---------------------------------------------------------------------------
# 6. Evaluation metric tests
# ---------------------------------------------------------------------------

def test_genre_diversity_all_same():
    """Diversity is 1/N when all same genre."""
    fake_recs = [
        Recommendation(song=SAMPLE_SONGS[0], hybrid_score=1.0, content_score=1.0,
                       collaborative_score=0.0, confidence=0.9, explanation="test"),
        Recommendation(song=SAMPLE_SONGS[0], hybrid_score=0.9, content_score=0.9,
                       collaborative_score=0.0, confidence=0.8, explanation="test"),
    ]
    diversity = evaluate_genre_diversity(fake_recs)
    assert diversity == 0.5  # 1 unique genre / 2 recs


def test_avg_confidence_empty():
    """Avg confidence of empty list is 0."""
    assert evaluate_avg_confidence([]) == 0.0


def test_precision_at_k():
    """Precision@K returns a value in [0, 1]."""
    rec = HybridRecommender(SAMPLE_SONGS, SAMPLE_RATINGS)
    pop_ids = [s.id for s in SAMPLE_SONGS if s.genre == "pop"]
    precision = evaluate_precision_at_k(rec, POP_USER, pop_ids, k=3)
    assert 0.0 <= precision <= 1.0
