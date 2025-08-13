# AskReddit Project Report

> **Author:** Bob
> **zID:** z123456

**Marking**
- Content: Novel insights made concerning experiments and results.
- Structure: Logically organized into a coherent, easy to understand argument.
- Analysis: Novel interpretations of results supported by evidence.
- Presentation: Engaging style with clear explanations throughout.

## 1. Results
### 1.1 Commercial System Analysis: Reddit's Recommendation Engine
The primary commercial system for our recommendation scenario is Reddit's own multifaceted recommendation engine. It is designed to keep millions of users engaged by personalizing the content they see across the platform. Understanding its strengths and limitations is key to motivating our own project's proposal.

#### 1.1.1 Mechanisms to recommend content

The personalized home feed is the main touchpoint for users. The default "Best" sort is a complex algorithm that prioritizes fresh content from communities a user frequently interacts with. It considers not only post scores and age but also the user's historical engagement within those subreddits.

Besides, features like the "Explore" tab and notifications for trending posts are designed to introduce users to new content and communities that are algorithmically determined to be relevant to their interests.

Personalized content is not the only source of recommendation. For example, `r/popular` and `r/all` serve as global, non-personalized feeds that showcase the most popular content across the entire platform, providing a broad view of what is currently trending.

#### 1.1.2 Strengths
Reddit's system has several undeniable strengths:
- Scalability and Engagement: It operates effectively at a massive scale, successfully serving real-time recommendations to millions of users and fostering high levels of engagement within established communities.
- Community-Centric Model: The "Best" sort is extremely effective at reinforcing a user's existing interests by keeping them up-to-date with the communities they already love.

#### 1.1.3 Weaknesses and Motivation for Our Project
Despite its strengths, Reddit's system has inherent weaknesses that directly motivate the approach taken in our project:

- The "Filter Bubble" Effect: By heavily prioritizing content from a user's subscribed subreddits, the system can inadvertently limit discovery. It excels at showing a user more of what they already know, but it is less effective at helping them discover new, thematically similar content from communities they don't follow.
- Limited Granularity Within Subreddits: Reddit's recommendations are often driven by community-level signals rather than the specific content of individual posts. A user might upvote a philosophical post in a large, diverse subreddit like r/Showerthoughts, but the algorithm might then recommend a popular but unrelated pun from the same community simply because it's popular. It struggles to differentiate between the varied content within a single subreddit.

Our project was designed to address these specific limitations. By focusing on purely content-based filtering, we aimed to build a system that recommends posts based on what they are about, not just where they were posted. Our core proposal was to see if we could break the "filter bubble" by identifying thematically similar posts regardless of their subreddit of origin.

By focusing on a single, highly diverse subreddit like r/Showerthoughts, we created a perfect test case to evaluate if our content-based models (like LLM embeddings and Vector Negation) could achieve a more granular understanding of user preferences than a purely community-based model. Our goal was to prove the value of a content-first approach in a challenging real-world scenario.

### 1.2 Summary of Approaches

#### 1.2.1 Datasets
Our project utilized the "Huge Collection of Reddit Votes" dataset available on Kaggle. This dataset is composed of two primary files: a vote log containing over 44 million user votes and a submissions file detailing over 22 million posts across 139,000 subreddits.

Given the project's scope, we made the strategic decision to narrow our focus to a single, highly active community: r/Showerthoughts. This decision allowed us to perform a deep, content-focused analysis without the confounding variable of cross-community user behavior.

#### 1.2.2 Preprocessing and Final Dataset

We filtered the master submissions file to isolate all posts belonging to r/Showerthoughts, resulting in a corpus of 95,084 unique posts after removing entries with no title.

We then filtered the 44_million_vote.txt file to extract all historical upvotes corresponding to these posts.

For our final evaluation, we identified the top 20 users with the highest number of upvotes within our dataset to ensure our analysis was based on the platform's most engaged members.

The primary strength of this dataset is its scale and authenticity, providing real-world user interaction data. Its main weakness, particularly for content-based filtering, is the idiosyncratic and diverse nature of the content, which presents a significant challenge in creating coherent user taste profiles.

#### 1.2.3 Methodologies
To address the recommendation problem, our team developed and evaluated a suite of models. For evaluation, all methods produced a ranked list of 200 recommendations.

##### 1.2.3.1 TF-IDF Vectorizer (Baseline)
This classic approach served as our strong baseline to measure the effectiveness of keyword-matching.

- Implementation/Library: We used the TfidfVectorizer from Python's scikit-learn library.
- Preprocessing: The title and selftext for each post were concatenated into a single text field.
- Parameters: The vectorizer was configured with the following parameters: `stop_words='english'` to remove common words, `max_df=0.95` to ignore terms appearing in more than 95% of documents, and `min_df=2` to ignore very rare terms.
- User Profiling & Ranking: The user profile was built using the same time-decay weighted average method as the LLM, and ranking was also performed using cosine similarity.

##### 1.2.3.2 Multilayer Perceptron (MLP) Neural Network

##### 1.2.3.3 Large Language Model (LLM) Embedding with Semantic Similarity

This approach was designed to move beyond simple keyword matching and understand the deep semantic meaning of the content. The core idea is to represent each post as a dense vector in a high-dimensional space where proximity indicates semantic similarity.

![LLM Embedding Workflow Chart](img/llm-flowchart.svg)
*Workflow of Generating Embeddings*

**Implementation and Library**: The entire workflow was implemented in Python using the [sentence-transformers](https://sbert.net/) library, a framework built on top of PyTorch that simplifies the use of pre-trained models for embedding generation.

**Bidirectional Encoder Representations from Transformers**
Or BERT, was a landmark model that fundamentally changed how machines understand natural language. Before BERT, models were largely unidirectional, meaning they read text either from left-to-right or right-to-left. BERT's key innovation was its ability to learn from the entire context of a sentence at once. As this paper by Google AI language[^1] states, "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers." This bidirectional understanding allows it to grasp context and nuance in a way that was previously not possible.

[^1]: 2018, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, <https://doi.org/10.48550/arXiv.1810.04805>

**Masked Language Modeling**
Masked Language Modeling (MLM) is the pre-training technique introduced with BERT. Its primary innovation was to enable a model to learn from both left and right context simultaneously, creating a truly bidirectional understanding of language.

Instead of trying to predict the next word in a sentence, MLM randomly hides a certain percentage of words in the input text and then tasks the model with predicting only those hidden words. As stated in this paper[^1], the objective is to "predict the original vocabulary id of the masked word based only on its context."

To prevent the model from simply learning to focus on the `[MASK]` token, BERT uses a strategy for the 15% of words it randomly chooses to hide:
- 80% of the time, the word is replaced with the `[MASK]` token (my dog is hairy → my dog is `[MASK]`).
- 10% of the time, the word is replaced with a random word (my dog is hairy → my dog is apple).
- 10% of the time, the word is left unchanged (my dog is hairy → my dog is hairy).

This forces the model to maintain a rich contextual understanding of every word, as it never knows which one it will be asked to predict.

**Permuted Language Modeling**
Permuted Language Modeling (PLM) was introduced with XLNet to address a key limitation of MLM. While MLM learns from bidirectional context, it assumes that each masked word is predicted independently of the others.

PLM introduces an autoregressive approach, similar to traditional language models, but on a shuffled or "permuted" version of the sentence. For a given permutation of a sentence, the model predicts the words one by one, but in the new shuffled order.

This solves MLM's independence assumption. As this paper about MPNet[^2] explains, "PLM factorizes the predicted tokens with the product rule in any permuted order... which avoids the independence assumption in MLM and can better model dependency among predicted tokens." For example, if the model has to predict "New" and "York", PLM allows the model to use its prediction of "New" to help it predict "York".

[^2]: 2020, MPNet: Masked and Permuted Pre-training for Language Understanding, <https://doi.org/10.48550/arXiv.2004.09297>

The main weakness of PLM is that during its autoregressive prediction, it doesn't know the original positions of the words that come later in the permuted sequence. This creates a "discrepancy between pre-training and fine-tuning," because during fine-tuning, the model always sees the full, un-permuted sentence.

![Unified View of MLM and PLM](img/mlm_plm.png)
*Note. A unified view of MLM and PLM, where x_i and p_i represent token and position embeddings. The left side in both MLM (a) and PLM (b) are in original order, while the right side in both MLM (a) and PLM (b) are in permuted order and are regarded as the unified view. From "MPNet: Masked and Permuted Pre-training for Language Understanding," by Kaitao Song, 2020, Nanjing University of Science and Technology*

**Neural Network Architecture and Training**: We selected the all-mpnet-base-v2 model, a high-performance sentence-transformer variant. This model is based on the MPNet architecture, which itself is a novel pre-training method that leverages the strengths of both Masked Language Modeling (like BERT) and Permuted Language Modeling (like XLNet). This hybrid approach allows the model to capture a rich understanding of bidirectional context.
The model was pre-trained on a massive dataset of over 1 billion text pairs from a diverse range of sources, including a large portion of Reddit comments. It was then fine-tuned using a contrastive learning objective, where the model learns to pull semantically similar sentences closer together in the vector space while pushing dissimilar ones apart. This training process makes it highly effective for generating meaningful sentence embeddings, and it achieves state-of-the-art performance on numerous sentence similarity and semantic search benchmarks. The model outputs a dense vector of 768 dimensions for each input text.

**Preprocessing**: For each post, the title and selftext were extracted. Since titles are often more concise and representative of a post's core idea, we created a single representative vector for each post by calculating a weighted average of the two embeddings. The title embedding was given a 60% weight, and the selftext embedding was given a 40% weight. This combined vector was then used for all subsequent calculations.

**User Profiling**: To model a user's taste, we constructed a profile vector from their historical upvotes. To account for evolving interests, we implemented a time-decay weighted average. Using a decay_rate of 0.95, this method gives exponentially more weight to recent upvotes, ensuring the user profile is more reflective of their current tastes rather than being a simple average of all past interactions.

**Ranking Function**: Recommendations were generated by ranking all candidate posts based on their similarity to the user's profile. We used Cosine Similarity as our similarity function, which measures the cosine of the angle between the user profile vector and each post vector. A higher cosine similarity score (closer to 1.0) indicates a stronger semantic match. The top 200 posts with the highest similarity scores were selected as the final recommendations for evaluation.

##### 1.2.3.4 Support Vector Machine

##### 1.2.3.5 Vector Negation

##### 1.2.3.6Sentence-BERT and LightFM

### 1.3 Evaluation
> final proposed recommender

