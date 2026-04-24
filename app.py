"""
Streamlit frontend for the Hybrid Music Recommender.
Run with: streamlit run app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import (
    load_songs, load_ratings, HybridRecommender, UserProfile,
    evaluate_genre_diversity, evaluate_avg_confidence, Recommendation,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TuneRanker", page_icon="🎵", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    .main .block-container { max-width: 1100px; padding-top: 2rem; }

    h1, h2, h3 { font-family: 'Poppins', sans-serif; }

    .song-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%);
        border: 1px solid #3a3a5c;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    .song-card h4 {
        margin: 0 0 0.3rem 0;
        color: #a78bfa;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
    }
    .song-card .artist { color: #94a3b8; font-size: 0.9rem; }
    .song-card .tags span {
        display: inline-block;
        background: #3b3b5c;
        color: #c4b5fd;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 6px;
    }
    .song-card .explanation {
        margin-top: 0.6rem;
        font-size: 0.85rem;
        color: #a5b4c8;
        font-style: italic;
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: #2a2a4a;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
    }
    .metric-box {
        background: #1e1e2f;
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: #e0e0e0;
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
        font-family: 'Poppins', sans-serif;
    }
    .metric-box .label { font-size: 0.8rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_recommender():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    songs = load_songs(os.path.join(data_dir, "songs.csv"))
    ratings = load_ratings(os.path.join(data_dir, "user_ratings.csv"))
    rec = HybridRecommender(songs, ratings)
    return rec, songs


recommender, all_songs = init_recommender()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🎵 TuneRanker")
st.markdown("*A hybrid music recommendation engine powered by scikit-learn*")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — User Preferences
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎧 Your Preferences")

    genre = st.selectbox("Favorite Genre", recommender.get_available_genres())
    mood = st.selectbox("Favorite Mood", recommender.get_available_moods())
    energy = st.slider("Target Energy", 0.0, 1.0, 0.7, 0.05)
    likes_acoustic = st.toggle("Prefer Acoustic", value=False)

    st.markdown("---")
    st.header("❤️ Songs You Like")
    st.caption("Select songs you already enjoy — this powers collaborative filtering.")
    liked = st.multiselect(
        "Pick some favorites",
        options=[(s.id, f"{s.title} — {s.artist}") for s in all_songs],
        format_func=lambda x: x[1],
        default=[],
    )
    liked_ids = [sid for sid, _ in liked]

    st.markdown("---")
    st.header("⚙️ Engine Settings")
    k = st.slider("Number of recommendations", 3, 15, 5)
    content_w = st.slider("Content weight", 0.0, 1.0, 0.6, 0.05)
    collab_w = round(1.0 - content_w, 2)
    st.caption(f"Collaborative weight: {collab_w}")

    recommend_btn = st.button("🎶 Get Recommendations", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if recommend_btn:
    # Update weights
    recommender.content_weight = content_w
    recommender.collab_weight = collab_w

    user = UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
    )

    with st.spinner("Finding your perfect tracks..."):
        recs = recommender.recommend(user, liked_song_ids=liked_ids, k=k)

    if not recs:
        st.warning("No recommendations found. Try adjusting your preferences.")
    else:
        # --- Metrics row ---
        diversity = evaluate_genre_diversity(recs)
        avg_conf = evaluate_avg_confidence(recs)
        top_score = recs[0].hybrid_score if recs else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{avg_conf:.0%}</div>
                <div class="label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{diversity:.0%}</div>
                <div class="label">Genre Diversity</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{top_score:.2f}</div>
                <div class="label">Top Hybrid Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Your Recommendations")

        # --- Song cards ---
        for i, r in enumerate(recs, 1):
            conf_pct = r.confidence * 100
            st.markdown(f"""
            <div class="song-card">
                <h4>#{i} · {r.song.title}</h4>
                <div class="artist">{r.song.artist}</div>
                <div class="tags">
                    <span>{r.song.genre}</span>
                    <span>{r.song.mood}</span>
                    <span>energy {r.song.energy:.2f}</span>
                    <span>tempo {r.song.tempo_bpm:.0f} bpm</span>
                </div>
                <div class="explanation">💡 {r.explanation}</div>
                <div style="display:flex; justify-content:space-between; margin-top:0.5rem; font-size:0.78rem; color:#94a3b8;">
                    <span>Content: {r.content_score:.3f}</span>
                    <span>Collab: {r.collaborative_score:.3f}</span>
                    <span>Hybrid: {r.hybrid_score:.3f}</span>
                    <span>Confidence: {conf_pct:.0f}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{conf_pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Score breakdown expander ---
        with st.expander("📊 Detailed Score Breakdown"):
            import pandas as pd
            df = pd.DataFrame([{
                "Song": r.song.title,
                "Artist": r.song.artist,
                "Genre": r.song.genre,
                "Mood": r.song.mood,
                "Content Score": r.content_score,
                "Collab Score": r.collaborative_score,
                "Hybrid Score": r.hybrid_score,
                "Confidence": f"{r.confidence:.0%}",
            } for r in recs])
            st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # Landing state
    st.info("👈 Set your preferences in the sidebar and click **Get Recommendations** to start.")

    with st.expander("ℹ️ How it works"):
        st.markdown("""
**TuneRanker** uses a hybrid recommendation approach combining two techniques:

**Content-Based Filtering** analyzes audio features (energy, tempo, valence,
danceability, acousticness) using scikit-learn's KNN algorithm to find songs
that sound similar to your preferences.

**Collaborative Filtering** looks at what other users with similar taste
enjoyed, using a user-item rating matrix and NearestNeighbors to surface
songs you might have missed.

The two scores are blended with configurable weights to produce a final
hybrid score. Each recommendation includes a confidence rating and a
plain-English explanation of why it was chosen.
        """)

    with st.expander("📂 Catalog Overview"):
        import pandas as pd
        df = pd.DataFrame([{
            "Title": s.title, "Artist": s.artist, "Genre": s.genre,
            "Mood": s.mood, "Energy": s.energy,
        } for s in all_songs])
        st.dataframe(df, use_container_width=True, hide_index=True)
