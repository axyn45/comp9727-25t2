# AskReddit Project Report

> **Author:** Bob
> **zID:** z123456

**Marking**
- Content: Novel insights made concerning experiments and results.
- Structure: Logically organized into a coherent, easy to understand argument.
- Analysis: Novel interpretations of results supported by evidence.
- Presentation: Engaging style with clear explanations throughout.
## Results
### Commercial System Analysis: Reddit's Recommendation Engine
The primary commercial system for our recommendation scenario is Reddit's own multifaceted recommendation engine. It is designed to keep millions of users engaged by personalizing the content they see across the platform. Understanding its strengths and limitations is key to motivating our own project's proposal.

#### Mechanisms to recommend content

The personalized home feed is the main touchpoint for users. The default "Best" sort is a complex algorithm that prioritizes fresh content from communities a user frequently interacts with. It considers not only post scores and age but also the user's historical engagement within those subreddits.

Besides, features like the "Explore" tab and notifications for trending posts are designed to introduce users to new content and communities that are algorithmically determined to be relevant to their interests.

Personalized content is not the only source of recommendation. For example, `r/popular` and `r/all` serve as global, non-personalized feeds that showcase the most popular content across the entire platform, providing a broad view of what is currently trending.

#### Strengths
Reddit's system has several undeniable strengths:
- Scalability and Engagement: It operates effectively at a massive scale, successfully serving real-time recommendations to millions of users and fostering high levels of engagement within established communities.
- Community-Centric Model: The "Best" sort is extremely effective at reinforcing a user's existing interests by keeping them up-to-date with the communities they already love.

#### Weaknesses and Motivation for Our Project
Despite its strengths, Reddit's system has inherent weaknesses that directly motivate the approach taken in our project:

- The "Filter Bubble" Effect: By heavily prioritizing content from a user's subscribed subreddits, the system can inadvertently limit discovery. It excels at showing a user more of what they already know, but it is less effective at helping them discover new, thematically similar content from communities they don't follow.
- Limited Granularity Within Subreddits: Reddit's recommendations are often driven by community-level signals rather than the specific content of individual posts. A user might upvote a philosophical post in a large, diverse subreddit like r/Showerthoughts, but the algorithm might then recommend a popular but unrelated pun from the same community simply because it's popular. It struggles to differentiate between the varied content within a single subreddit.

Our project was designed to address these specific limitations. By focusing on purely content-based filtering, we aimed to build a system that recommends posts based on what they are about, not just where they were posted. Our core proposal was to see if we could break the "filter bubble" by identifying thematically similar posts regardless of their subreddit of origin.

By focusing on a single, highly diverse subreddit like r/Showerthoughts, we created a perfect test case to evaluate if our content-based models (like LLM embeddings and Vector Negation) could achieve a more granular understanding of user preferences than a purely community-based model. Our goal was to prove the value of a content-first approach in a challenging real-world scenario.

### Summary of Approaches

#### Datasets
Our project utilized the "Huge Collection of Reddit Votes" dataset available on Kaggle. This dataset is composed of two primary files: a vote log containing over 44 million user votes and a submissions file detailing over 22 million posts across 139,000 subreddits.

Given the project's scope, we made the strategic decision to narrow our focus to a single, highly active community: r/Showerthoughts. This decision allowed us to perform a deep, content-focused analysis without the confounding variable of cross-community user behavior.

#### Preprocessing and Final Dataset

We filtered the master submissions file to isolate all posts belonging to r/Showerthoughts, resulting in a corpus of 95,084 unique posts after removing entries with no title.

We then filtered the 44_million_vote.txt file to extract all historical upvotes corresponding to these posts.

For our final evaluation, we identified the top 20 users with the highest number of upvotes within our dataset to ensure our analysis was based on the platform's most engaged members.

The primary strength of this dataset is its scale and authenticity, providing real-world user interaction data. Its main weakness, particularly for content-based filtering, is the idiosyncratic and diverse nature of the content, which presents a significant challenge in creating coherent user taste profiles.

#### Methodologies
To address the recommendation problem, our team developed and evaluated a suite of models. For evaluation, all methods produced a ranked list of 200 recommendations.

##### TF-IDF Vectorizer (Baseline)
This classic approach served as our strong baseline to measure the effectiveness of keyword-matching.

- Implementation/Library: We used the TfidfVectorizer from Python's scikit-learn library.
- Preprocessing: The title and selftext for each post were concatenated into a single text field.
- Parameters: The vectorizer was configured with the following parameters: `stop_words='english'` to remove common words, `max_df=0.95` to ignore terms appearing in more than 95% of documents, and `min_df=2` to ignore very rare terms.
- User Profiling & Ranking: The user profile was built using the same time-decay weighted average method as the LLM, and ranking was also performed using cosine similarity.

##### Multilayer Perceptron (MLP) Neural Network

##### Large Language Model (LLM) Embedding with Semantic Similarity
This approach was designed to move beyond simple keyword matching and understand the deep semantic meaning of the content.

Implementation/Library: We used the sentence-transformers library in Python.

Model: The all-mpnet-base-v2 model was chosen for its high performance in generating sentence embeddings. This model produces a 768-dimensional vector for each piece of text.

Preprocessing: The title and selftext of each post were extracted. To create a single representative vector, we calculated a weighted average of the two embeddings (80% weight for the title, 20% for the selftext).

User Profiling: A user's taste profile was constructed by calculating a time-decay weighted average of the embeddings of posts they had previously upvoted. This method, using a decay_rate of 0.95, gives more importance to recent interactions.

Ranking: Recommendations were generated by using Cosine Similarity to find and rank the posts most semantically similar to the user's calculated taste profile.

##### Support Vector Machine

##### Vector Negation

##### Sentence-BERT and LightFM

### Evaluation
> final proposed recommender

