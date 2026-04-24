# 🎧 Model Card: TuneRanker — Hybrid Music Recommender

## 1. Model Name

**TuneRanker 2.0** — a hybrid music recommendation engine combining content-based and collaborative filtering.

---

## 2. Intended Use

This system suggests 3–15 songs from a 100-song catalog based on a user's preferred genre, mood, energy level, acoustic preference, and listening history. It is designed as a classroom and portfolio project demonstrating hybrid recommendation with reliability testing. It is not intended for production use with real users or real streaming data.

---

## 3. How It Works

TuneRanker uses two recommendation strategies that work together:

**Content-based filtering** looks at the audio features of songs — things like energy, tempo, how danceable they are, and how acoustic they sound. It uses a machine learning technique called K-Nearest Neighbors (KNN) to find songs in the catalog whose features are closest to what the user says they like. It also gives bonus points if a song's genre or mood matches exactly.

**Collaborative filtering** looks at patterns in how other users rated songs. If User A and User B both liked the same three songs, and User B also liked a fourth song, the system might recommend that fourth song to User A. It builds a table of all users and all songs, then uses NearestNeighbors to find which songs have similar rating patterns to the ones you already like.

The two scores are combined using a weighted average (default: 60% content, 40% collaborative) to produce a final hybrid score. Each recommendation also gets a confidence rating (0–100%) and a plain-English explanation of why it was chosen.

---

## 4. Data

The dataset contains **100 songs** across 10 genres (pop, lofi, rock, jazz, ambient, edm, synthwave, indie pop, folk, and folk) and 7 moods (happy, chill, intense, relaxed, focused, moody). Each song has 5 numerical audio features: energy, tempo_bpm, valence, danceability, and acousticness.

The **user ratings** dataset contains **200 ratings** from **20 simulated users**, each rating 10 songs on a 1–5 scale. Users were designed to have distinct taste profiles (e.g., User 1 likes pop, User 2 likes lofi, User 3 likes rock).

Both datasets are synthetic. The songs use fictional artist names and titles. The data was designed to cover a range of genres and moods but reflects a Western-centric, English-language music perspective. Genres like K-pop, reggaeton, Afrobeats, Bollywood, and classical are not represented.

---

## 5. Strengths

- Works well for users with clear, consistent preferences (e.g., "I want happy pop" or "I want chill lofi"). The top recommendations consistently match the requested genre and mood.
- Every recommendation is explainable — users can see exactly why a song was chosen and how much each engine contributed to the score.
- Confidence scores help users gauge how strong a match is. High-confidence recommendations (90%+) are reliably on-target.
- The hybrid approach catches songs that pure content-based filtering would miss — collaborative filtering can surface cross-genre discoveries.
- The system is transparent and simple enough to debug. When a recommendation seems wrong, you can trace it back to the specific scores.

---

## 6. Limitations and Bias

**Technical limitations:**
- The 100-song catalog is tiny. A real recommender needs thousands or millions of songs to be useful. With only ~10 songs per genre, the system has very little to choose from.
- 20 simulated users produce a very sparse rating matrix. Collaborative filtering needs dense data to find meaningful patterns — with this little data, it mostly reinforces content-based results rather than adding new signal.
- The system has no feedback loop. It can't learn from whether a user actually liked a recommendation.
- Audio features are static numbers — the system doesn't understand lyrics, cultural context, language, or the emotional arc of a song.

**Bias concerns:**
- Genre bias: The catalog is skewed toward Western genres. Users who prefer genres not in the catalog (classical, hip-hop, Latin, K-pop) get poor recommendations.
- Popularity bias: Collaborative filtering naturally favors songs that many users have rated. Niche songs with few ratings get recommended less, even if they'd be a great match.
- Mood oversimplification: Moods like "happy" and "chill" are reductive. A song can be bittersweet, nostalgic, or energetically complex — none of which this system captures.
- The simulated user profiles were designed by one person, so the collaborative filtering patterns reflect that person's assumptions about how different "types" of listeners behave.

**If used in a real product**, these biases could create filter bubbles where users only hear more of what they already know, reinforcing existing taste rather than broadening it.

---

## 7. Evaluation

**Automated testing:** 16 unit tests covering data loading, both engines, hybrid blending, confidence scoring, and evaluation metrics. All 16 pass.

**Quantitative metrics:**
- Precision@5 for a pop user against pop songs: 1.00 (perfect — all 5 recs were pop)
- Genre diversity for a pop user: 0.20 (1 unique genre in 5 recs — expected, since the user wants pop)
- Average confidence across top-5: 0.975
- The system was tested with 3 distinct user profiles (pop/happy, lofi/chill, rock/intense) and produced genre-appropriate results for all three.

**Human evaluation:** I manually reviewed the top-5 recommendations for each test profile and confirmed they matched my expectations. The rock/intense profile was the most accurate — every recommendation was a high-energy rock song by Voltline. The pop/happy profile occasionally included songs that were technically pop but felt more "party" than "happy," suggesting the mood labels could be more granular.

**What surprised me:** Small changes to the content/collaborative weight ratio produced large changes in output. Moving from 60/40 to 80/20 eliminated most collaborative signal. This showed me that hybrid systems are sensitive to tuning and need careful calibration.

---

## 8. Future Work

- Expand the catalog to 1,000+ real songs using a public dataset (e.g., Spotify features dataset from Kaggle)
- Replace simulated users with real anonymized listening data
- Add diversity constraints so the system doesn't recommend 5 songs by the same artist
- Implement a feedback loop where users can thumbs-up/down recommendations to improve future suggestions
- Add tempo as a user preference (currently only used as a feature, not a preference)
- Explore matrix factorization (SVD) as an alternative to KNN-based collaborative filtering

---

## 9. Ethical Reflection

### What are the limitations or biases in your system?

The biggest bias is cultural. The catalog only contains Western genres, and the simulated user profiles assume Western listening patterns. If deployed for a global audience, this system would systematically underserve users from non-Western musical traditions. Additionally, collaborative filtering amplifies majority preferences — if most users in the ratings data like pop, pop songs get recommended disproportionately even to users who might prefer something else.

### Could your AI be misused, and how would you prevent that?

A recommendation system like this could be misused to manipulate listening behavior — for example, a platform could boost songs from artists who pay for promotion without disclosing it. To prevent this, recommendations should be transparent (TuneRanker already explains every recommendation) and the scoring algorithm should be auditable. In a real product, I'd add disclosure when any non-organic signal (paid promotion, editorial curation) affects rankings.

### What surprised you while testing your AI's reliability?

I was surprised by how brittle the collaborative engine is with sparse data. With only 20 users and 200 ratings, many songs have no collaborative signal at all, so the hybrid score collapses to pure content-based for those items. This taught me that collaborative filtering needs critical mass to be useful — you can't just add it and expect it to help with a small dataset.

### Describe your collaboration with AI during this project.

I used Claude (Anthropic) extensively during development. 

**Helpful suggestion:** When I was designing the confidence scoring, Claude suggested normalizing hybrid scores relative to the min/max of the current result set rather than using absolute thresholds. This was a good idea because it means confidence is always interpretable on a 0–100% scale regardless of how the weights are configured.

**Flawed suggestion:** Early in development, Claude suggested using TF-IDF on genre/mood strings as features for the content engine. This was unnecessary and overcomplicated — genre and mood are categorical labels with only 10 and 7 unique values respectively. Simple bonus scoring (exact match = +0.25, partial match = +0.12) was more interpretable and worked just as well. I learned that not every suggestion from AI is the right fit for the problem scale.
