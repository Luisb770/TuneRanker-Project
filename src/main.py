"""
Command-line runner for the Music Recommender.
Demonstrates the hybrid engine with sample user profiles.
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.recommender import (
    load_songs, load_ratings, HybridRecommender, UserProfile,
    evaluate_precision_at_k, evaluate_genre_diversity, evaluate_avg_confidence,
)


def main() -> None:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    songs = load_songs(os.path.join(data_dir, "songs.csv"))
    ratings = load_ratings(os.path.join(data_dir, "user_ratings.csv"))

    recommender = HybridRecommender(songs, ratings)

    # --- Example 1: Happy pop listener ---
    print("\n" + "=" * 60)
    print("  Example 1: Happy Pop Listener")
    print("=" * 60)
    user1 = UserProfile(favorite_genre="pop", favorite_mood="happy",
                        target_energy=0.8, likes_acoustic=False)
    recs1 = recommender.recommend(user1, liked_song_ids=[1, 5, 20], k=5)
    for r in recs1:
        print(f"\n  {r.song.title} by {r.song.artist}")
        print(f"  Genre: {r.song.genre} | Mood: {r.song.mood} | Energy: {r.song.energy}")
        print(f"  Hybrid Score: {r.hybrid_score:.3f} | Confidence: {r.confidence:.1%}")
        print(f"  {r.explanation}")

    # --- Example 2: Chill lofi listener ---
    print("\n" + "=" * 60)
    print("  Example 2: Chill Lofi Listener")
    print("=" * 60)
    user2 = UserProfile(favorite_genre="lofi", favorite_mood="chill",
                        target_energy=0.4, likes_acoustic=True)
    recs2 = recommender.recommend(user2, liked_song_ids=[2, 4, 9], k=5)
    for r in recs2:
        print(f"\n  {r.song.title} by {r.song.artist}")
        print(f"  Genre: {r.song.genre} | Mood: {r.song.mood} | Energy: {r.song.energy}")
        print(f"  Hybrid Score: {r.hybrid_score:.3f} | Confidence: {r.confidence:.1%}")
        print(f"  {r.explanation}")

    # --- Example 3: Intense rock listener ---
    print("\n" + "=" * 60)
    print("  Example 3: Intense Rock Listener")
    print("=" * 60)
    user3 = UserProfile(favorite_genre="rock", favorite_mood="intense",
                        target_energy=0.9, likes_acoustic=False)
    recs3 = recommender.recommend(user3, liked_song_ids=[3, 18, 28], k=5)
    for r in recs3:
        print(f"\n  {r.song.title} by {r.song.artist}")
        print(f"  Genre: {r.song.genre} | Mood: {r.song.mood} | Energy: {r.song.energy}")
        print(f"  Hybrid Score: {r.hybrid_score:.3f} | Confidence: {r.confidence:.1%}")
        print(f"  {r.explanation}")

    # --- Evaluation metrics ---
    print("\n" + "=" * 60)
    print("  Evaluation Metrics")
    print("=" * 60)
    pop_ids = [s.id for s in songs if s.genre == "pop"]
    p_at_5 = evaluate_precision_at_k(recommender, user1, pop_ids, k=5)
    print(f"  Precision@5 (pop user vs pop songs): {p_at_5:.2f}")

    diversity = evaluate_genre_diversity(recs1)
    print(f"  Genre Diversity (user 1 recs): {diversity:.2f}")

    avg_conf = evaluate_avg_confidence(recs1)
    print(f"  Avg Confidence (user 1 recs): {avg_conf:.3f}")


if __name__ == "__main__":
    main()
