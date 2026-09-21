# COMP9727 Recommender Systems

Coursework for COMP9727 (Recommender Systems, UNSW): an individual assignment, a team project,
and weekly tutorial notebooks.

## Assignment (30%): Content-Based Music Recommendation

Build a content-based music recommender over 1,500 songs labelled with one of five topics
(dark, emotion, lifestyle, personal, sadness), using artist, track, release date, genre and lyrics.

- **Part 1 — Topic classification:** compare Bernoulli and Multinomial Naive Bayes with SVM across
  preprocessing choices (text cleaning, stopword lists, stemming vs. lemmatisation) and vocabulary
  sizes, evaluated with cross-validated accuracy, macro/weighted F1 and confusion matrices.
  Best overall method: SVM with TF-IDF, 400 features.
- **Part 2 — Recommendation methods:** build per-topic tf-idf user profiles for three simulated
  users and match them against test-week songs (cosine similarity vs. Euclidean distance),
  evaluated with Precision/Recall/F1@20.
- **Part 3 — User study:** one participant works through a recommendation list using a talk-aloud
  protocol; human judgement scored far better than the simulated users (F1 0.75 vs ~0.15).

Code and write-up: [`assignment/main.ipynb`](assignment/main.ipynb).

## Project (50%): AskReddit — Content Recommender for Reddit

Team project (5 members) recommending posts from `r/Showerthoughts`, using Kaggle's *Huge Collection
of Reddit Votes* (44M votes, 22M posts) narrowed down to 95,084 posts and the 20 most active voters.
Six approaches were implemented and compared offline with **NDCG@200**:

| Approach | Core idea |
| --- | --- |
| TF-IDF (baseline) | keyword vectors + cosine similarity |
| LLM embeddings | `all-mpnet-base-v2` 768-d post vectors, time-decay weighted user profile |
| SVM | per-user upvote/downvote classifier over TF-IDF features |
| MLP | neural network predicting user upvotes |
| Vector Negation | keyword vectors minus the subspace of disliked content |
| Sentence-BERT | SBERT embeddings with hybrid and sequential extensions |

**Outcome:** Vector Negation performed best overall; LLM embeddings and TF-IDF were strong for
users with consistent taste; a two-stage retrieve-and-re-rank hybrid is proposed for real-world use.

- [`project/group-submission/`](project/group-submission) — team notebook, presentation, results
- [`project/report/`](project/report) — individual report on the LLM embedding pipeline and results
- [`project/results/`](project/results) — NDCG@200 comparison tables and plots
- [`project/share/`](project/share), [`project/testfield/`](project/testfield), [`project/aggregate/`](project/aggregate) — shared steps, experiments, aggregation
- [`project/project-design.md`](project/project-design.md) — initial design proposal ("Gamer's Partner", a hybrid Steam recommender)

## Tutorials

Weekly notebook exercises: [`week01/`](week01) – [`week04/`](week04).

## Notes

Datasets are excluded (per submission rules); the embedding `.npz` files are tracked with Git LFS.
