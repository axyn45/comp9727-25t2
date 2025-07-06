# Project: Gamer's Partner

> Author: ---
> zID: ---

In an era of abundant digital entertainment, players often face the paradox of choice, struggling to find new games that align with their unique tastes. The sheer volume of available titles on platforms like Steam, Epic Games Store, and console marketplaces makes manual discovery a significant challenge. This document details the design of a sophisticated recommender system aimed at solving this problem. By intelligently analyzing player behavior, preferences, and the inherent attributes of games, our system will provide tailored recommendations that connect players with their next favorite game. The core objectives are to increase player satisfaction, drive engagement, and create a more personalized and compelling user experience.

## 1 Scope

- [x] what's the domain and target user
- [x] how many items presented and through what way
- [x] simulate user input
- [x] address updating model and cold start
- [x] business consideration

### 1.1 Project Domain

This project operates within the video game and digital entertainment industry. The system is designed to function within a digital storefront or platform where games are sold and distributed (e.g., Steam, Epic Games Store, GOG, or a console marketplace like the PlayStation Store). The primary role is to drive user activity and purchases within this commercial environment.

Beyond profit consideration, the system also focus on enhancing the user experience. By making discovery easier and more personal, it aims to keep players invested in the platform, encouraging them to return, play more, and explore a wider variety of the catalog.

### 1.2 Target User

This system targets the players on game distribution platforms. Essentially we can categorize our intended users into these two groups:

1. **Newcomer** who is new to the platform or to gaming in general. They are often overwhelmed by a large catalog and need clear, accessible recommendations.

2. **Explorer** who enjoys discovering unique, niche, or indie titles that fall outside of the mainstream best-sellers. They value recommendations that are surprising and align with their specific, often eclectic, tastes. The recommender system helps them sift through thousands of titles to find these games.

### 1.3 Delivery of Recommendation

When a user requests recommendations, the system will generate several candidate sets:

- Collaborative Candidates: Based on the user's embedding, find games whose embeddings are most similar. This provides personalized, "serendipitous" recommendations.
- Content-Based Candidates: Based on a user's recently played or viewed game, find games with similar metadata. This provides transparent, "similar to X" recommendations.

Then the candidate sets are then combined into a single list. A machine learning model (like a simple logistic regression or a more complex model like Gradient Boosted Trees) can be used to rank this list. Features for this ranking model would include:

- The collaborative filtering score
- The content-based similarity score
- The game's global popularity (from playtime data)
- The game's community sentiment (from reviews)
- The game's novelty or release date

We're planning to build a Web UI that allows gamers to interact with the system[^1] and access the recommendations. Gamers will be presented with 10-20 games as tiles on the webpage (refresh the page to get more recommendations), each with basic information displayed like game title, category/tags and overall rating (displayed as scores, more on that later). By clicking on the tile, user can see the full description and reviews (if applicable) of the game.

[^1]: This is not guranteed and may be replaced by a simulated solution.

### 1.4 Simulating User Inputs

There are two major types of user interaction we are considering to use for training:

**Explicit Interaction**

1. Ratings/Reviews: The star ratings or thumbs up/down are the strongest explicit signals. This data (from the mock australian_user_reviews.json) directly informs the user's profile and can be used to fine-tune the recommendation models.

2. Wishlist: Adding a game to a wishlist is a powerful signal of future purchase intent. It can be used to tune recommendations towards items the user is already considering.

**Implicit Feedback**

1. Playtime: This is the most important implicit signal. High playtime on a game strongly reinforces a user's preference for its genre, tags, and general characteristics. The system should weigh playtime heavily when updating user profiles.

2. Clicks and Views: Clicking on a recommended game and spending time on its page is a positive signal. Ignoring a recommendation is a weak negative signal. This data is used to re-rank recommendations in near real-time.

### 1.5 Updating Model

Updating a recommendation model in a live production environment requires a dynamic architecture that can react to user interactions in real-time. Instead of periodically retraining a static model from scratch, we need to adopt a system that continuously learns and adapts.

**Possible Strategies**

1. **Incremental Training**: The core model itself is designed to be updated incrementally in this approach. Instead of being completely retrained, the model's parameters are adjusted with each new piece of data or with small mini-batches of recent data. As the pipline receives new user interactions, it transforms them into feature vectors. These vectors are then fed into the model's `partial_fit` (from scikit-learn) or equivalent function, which updates the model's weights without starting from scratch[^2].
2. **Rapid, Automated Batch Retraining**: For some models that cannot be updated incrementally, the solution is to drastically shorten the retraining cycle. The system automatically triggers a full model retraining process on a much more frequent basis, such as every hour. This is made possible by a highly automated CI/CD (Continuous Integration/Continuous Deployment) pipeline for machine learning, often called "MLOps"[^3].

[^2]: [What is Incremental Learning?](https://www.datacamp.com/blog/what-is-incremental-learning)

[^3]: [Retraining Model During Deployment: Continuous Training and Continuous Testing](https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing)

### 1.6 Addressing Cold Start

In this system we can break down the cold start problem into 2 parts:

- **New User Cold Start**: A new player signs up. The system knows nothing about their tastes and cannot provide personalized recommendations.
- **New Item Cold Start**: A new game is released. It has no ratings or play history, so the system doesn't know who to recommend it to.

To simplify the question, **we'll only consider the cold start problem for new user registrations**, assuming no further titles will be added to the game library. Note that this is not realistic in a production environment, but doing so allows us to focus and put effort on at least one real-world issue with limited time.

**Quick Start**

The goal is to gather preference data as quickly as possible during a user's first sessions. Now, instead of showing them a blank page, guide them through a brief onboarding process to gather basic informations of a user's interests.

We can present a visually appealing screen where the user can select their favorite genres (e.g., RPG, Strategy, FPS), themes (e.g., Sci-Fi, Fantasy, Horror), or gameplay mechanics (e.g., Open World, Crafting, Roguelike). This is a great opportunity to learn about the user at the beginning stage of recommendation.

**Initial Recommendations**

The data gathered from "Quick Start" phase is immediately used to generate the first set of recommendations. Content-based filtering is the primary tool for new users. For example, if a player selected "RPG" and "Fantasy," the system will instantly recommend games with those specific tags. This model doesn't need any user interaction history, only game metadata (tags, genre, developer, ratings, etc.).

### 1.7 Business Consideration

The fundamental business model for most digital game platforms is a commission on sales. The platform hosts games for developers and, in return, takes a percentage (typically 12-30%[^4]) of every game or in-game item sold. Therefore, the core financial objective is to maximize the total value of transactions.

The recommender system is one of the key driver of this objective. By showing the right game to the right player at the right time, the system dramatically increases the likelihood of a purchase. It moves users from "just Browse" to "adding to cart" by reducing the friction of discovery.

Also, a good recommender system makes the platform "stickier". When players feel the platform "gets them", they return more frequently, spend more time browsing. This increases the lifetime value[^5] of each customer, as now users are far more likely to make future purchases.

[^4]: [Platform Fees in the Videogame Industry: Full List](https://www.1d3.com/blog/platform-fees)

[^5]: [Lifetime value](https://www.optimizely.com/optimization-glossary/lifetime-value)

## 2 Datasets

- [x] describe the dataset
- [x] sufficient quality and quantity
- [x] which field is helpful in which way
- [x] limited breadth
- [ ] unrealistic data
- [x] overfit issue

### 2.1 Game Metadata and User Generated Content

[Recommender Systems and Personalization Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html) contains a collection of datasets that have been collected for research purposes by Julian McAuley, UCSD. We will use the collection of [Steam Video Game and Bundle Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) as our primary datasets. These datasets contain reviews from the Steam video game platform, and information about which games were bundled together.

In terms of this implementation, since we focus on building a hybrid system featuring content-based method along with collaborative filter, we will use item metadata as our primary training data source.

**Item Metadata**

Item metadata contains 32,135 entries of different games on Steam. Each record consists of these key data fields:

- Identifiers: `id`, `app_name`
- Content Features: `tags`, `genres`, `specs`, `developer`
- Quality/Popularity: `sentiment`, `reviews_url`
- Filtering Features: `price`, `discount_price`, `release_date`, `early_access`

```json
// Example
{
  'publisher':'Kotoshiro',
  'genres':[ 'Action', 'Casual', 'Indie', 'Simulation', 'Strategy'],
  'app_name':'Lost Summoner Kitty',
  'title':'Lost Summoner Kitty',
  'url':'http://store.steampowered.com/app/761140/Lost_Summoner_Kitty/',
  'release_date':'2018-01-04',
  'tags':['Strategy', 'Action', 'Indie', 'Casual', 'Simulation'],
  'discount_price':4.49,
  'reviews_url':'http://steamcommunity.com/app/761140/reviews/?browsefilter=mostrecent&p=1',
  'specs':['Single-player'],
  'price':4.99,
  'early_access':False,
  'id':'761140',
  'developer':'Kotoshiro'
}
```

**User Reviews**

User reviews (australian_user_reviews.json) contains 25,799 different reviews from australian users on Steam. Each record consists of these key data fields:

- Identifiers: `user_id`, `reviews>item_id`
- Sentiments: `recommend` (a boolean value indicating whether user would recommend this game to others), `helpful` (showing how many other users think this review is helpful/makes sense), `review` (the actual review texts)

```json
// Example
{
  'user_id':'ApxLGhost',
  'user_url':'http://steamcommunity.com/id/ApxLGhost',
  'reviews':[
    {
      'funny':'',
      'posted':'Posted December 14,
      2015.', 'last_edited':'',
      'item_id':'730',
      'helpful':'No ratings yet',
      'recommend':True,
      'review':'AMAZING GAME 10/10'
    }
  ]
}
```

**User-Item Data**

User-item data (australian_users_items.json) contains 88,310 records of user-item records from australian users on Steam. Each record shows the game library a user owns and how many hours one spent on each of the game in the library. Thry consist of these key data fields:

- Identifiers: `user_id`, `item_id`
- Content: `items` (an array listing all the games in the player's library)
- Implicit Rating: `playtime_forever` (indicating how much time a player spent on this game)

```json
// Example
{
  'user_id':'76561198089077856',
  'items_count':4,
  'steam_id':'76561198089077856',
  'user_url':'http://steamcommunity.com/profiles/76561198089077856',
  'items':[
    {
      'item_id':'205790',
      'item_name':'Dota 2 Test',
      'playtime_forever':0,
      'playtime_2weeks':0
    },
    {
      'item_id':'407250',
      'item_name':'Pro Evolution Soccer 2016 myClub',
      'playtime_forever':33,
      'playtime_2weeks':0
    },
    {
      'item_id':'466910',
      'item_name':'Worm.is: The Game',
      'playtime_forever':35,
      'playtime_2weeks':0
    },
    {
      'item_id':'485220',
      'item_name':'The Orb Chambers',
      'playtime_forever':1,
      'playtime_2weeks':0
    }
  ]
}
````


### 2.2 Fields Explained

**Building Game Profiles with Content-Based Features**

We will create a detailed "profile" or "feature vector" for every game using its metadata. This allows us to calculate a similarity score between any two games. For this, we will utlizing the following fields in the item metadata:

- `tags`: This is the most important field. It contains highly specific, user-generated descriptors like 'Open World', 'Crafting', 'Roguelike' and 'Atmospheric'. We would process this list of tags for each game using techniques like TF-IDF, to create a numerical vector that represents its unique characteristics.
- `genres`: This provides broader categories like 'Action', 'RPG', or 'Strategy'. This is a high-level feature that helps match games in the same category.
- `specs`: This field describes the game's features, such as 'Single-player', 'Multi-player', or 'Steam Achievements'. It adds another layer of detail for finding similar games. For example, a user who plays many 'Single-player' games can be recommended others with the same spec.
- `developer`: The developer is a very strong signal. Players who enjoy one game from a specific developer (e.g., 'CD PROJEKT RED' or 'Supergiant Games') are highly likely to enjoy their other titles, because games coming from same developer are likely to share similar genres and specs. We can treat this as a high-weight categorical feature.

**Quality Score from User Sentiment**

A game can be very similar in content to another but be of much lower quality. The sentiment field is the key to solving this. It provides aggregated community review scores like 'Overwhelmingly Positive', 'Mostly Positive', or 'Mixed'.

We would convert these categorical labels into a numerical score according to the design of Steam's review system[^6]. For example:

<center>

| Sentiment               | Score |
| ----------------------- | ----- |
| Overwhelmingly Positive | 10    |
| Very Positive           | 9     |
| Positive                | 7     |
| Mostly Positive         | 6     |
| Mixed                   | 5     |
| Mostly Negative         | 4     |
| Negative                | 3     |
| Very Negative           | 1     |
| Overwhelmingly Negative | 0     |

</center>

This allows the recommender to prioritize games that are not only similar in content but are also well-regarded by the community.

**Collaborative Filtering with Behavioral Data**

User-item data provides implicit user behavior data.

`user_id` and `item_id` are used to construct a massive user-item interaction matrix. This matrix is the fundamental data structure for collaborative filtering. The rows represent users, the columns represent games, and the values inside the matrix represent the strength of the interaction.

`playtime_forever` is the most valuable field in this dataset and will be the core value in a user-item matrix. It's a powerful implicit signal of engagement and preference. Raw playtime is heavily skewed. For example, some users play for thousands of hours, many play for less than one. To make it useful, we must transform it, typically using a logarithmic function like log(1 + playtime). This compresses the range and turns it into a more stable numerical score representing user preference. This transformed playtime score is the primary input for training a matrix factorization model (e.g., Alternating Least Squares - ALS). The model will learn to predict these playtime scores for games a user hasn't played yet, and the highest predictions become the recommendations.

The mere fact that a game is in a user's items list signifies a strong positive signal (that is the user have purchased it). A binary "owns/doesn't own" matrix can be used to build a simpler collaborative filtering model. The `items_count` can be helpful in identifying highly engaged users.

**Explicit User-Generated Content (UGC)**

User reviews provides explicit user feedback.

`recommend` is a powerful, high-quality, explicit signal. It's binary so we can easily tell if the user like it or not. A `True` value for `recommend` on a game strongly boosts that game's genres and tags within the user's preference profile. A `False` can be used to down-weight or create a negative profile of tags/genres the user dislikes. This boolean can be used as a "like" or "dislike" in various recommendation algorithms, providing clear guidance that complements the implicit playtime data. For instance, a game with high playtime but a False recommendation might be a guilty pleasure or a game the user now dislikes, a nuance that playtime alone cannot capture.

`review` (Text) is the most unstructured but potentially richest data source. Instead of just a binary recommend flag, we can apply NLP models to the review text to get a more detailed sentiment score (e.g., a score from -1.0 to 1.0). This can capture mixed feelings that the boolean misses. NLP helps to extract key topics or aspects from reviews. This allows us to build an incredibly detailed user profile. For example, a user who consistently mentions "great story" in their positive reviews can be recommended other story-rich games, even across different genres.

**Filtering**

Other fields in the dataset are essential for giving users control over their recommendations. For example, the recommender can include filters like "Free to Play", "Under $10", or "50% OFF" (by comparing `price` and `discount_price`).

Also users can filter to see "New Releases" or games from a specific era.

And `early_access` is a simple boolean filter to let users decide if they want to see games that are still in development.

With these filters, the system can be more efficient helping users to identify the games that interest them most.

**Generating Recommendations**

When a user expresses interest in a specific game (e.g., they played it, reviewed it, or are currently viewing its page), we use that game as our seed.

The system retrieves the feature vector for the seed game. It then compares this vector to all other games in the dataset and calculates a similarity score (e.g., using cosine similarity). This produces a ranked list of the most similar games based purely on content.

The system then takes this list of similar games and re-ranks it by multiplying the similarity score with the game's quality score (derived from sentiment).


[^6]: [Steam Review System](https://steamcommunity.com/discussions/forum/0/1744482869761322402/#c1744482869761428892)

### 2.3 Limitations

<!-- While the UCSD Steam dataset is fantastic for research and prototyping, using it to model a real-world system reveals several constraints. -->

**Incompleteness**

Each of these datasets is a snapshot, not the whole picture. It captures interactions up to a certain date and then stops, missing the continuous, dynamic flow of a live platform.

The datasets contains no information about games released after it was created. Your model will be completely blind to new hits, or major updates to existing games. Also,player preferences are not static. A genre that was niche when the data was collected could explode in popularity later (e.g., the rise of Auto-battlers or Extraction Shooters). The model cannot adapt to these evolving market trends.

A real-world platform like Steam should track a much wider array of user interactions than just playtime and reviews. This dataset is missing crucial signals that provide context and measure intent. Like for example, it lacks data on wishlist additions, which is a powerful signal of what a user wants to buy in the future. It doesn't capture clicks, page view duration, or trailer views. This short-term data is vital for session-based recommendations[^7].

<!-- **Unrealisticity** -->

**Overfitting**

This dataset was originally structured for a well-defined academic task: given a game's metadata, find or predict things about it. This encourages models that are optimized for a static "item-to-item" similarity task but may not be suitable for a dynamic production environment.

Because the datasets are (overall) rich in metadata and poor in user-item interactions, it heavily encourages the development of content-based filtering models. While useful, an over-reliance on this approach can lead to filter bubbles[^8] where users are only ever shown things that are very similar to what they already know.

[^7]: [Session-based Recommender Systems](https://session-based-recommenders.fastforwardlabs.com/)

[^8]: [How algorithms and filter bubbles decide what we see on social media](https://www.bbc.co.uk/bitesize/articles/zd9tt39)


## 3 Methods

- [ ] propose different methods
  - [ ] different types of RS
  - [ ] propose an approach
    - [ ] different methods or combined
- [ ] justify their suitability
  - [ ] evaluate methods and system

### 3.1 Type Variance and Justification

Given the goal of building a scalable, user-centric system, a multi-faceted approach is ideal. Here we are exploring different types of system including content-based, collaborative filtering, hybrid and knowledge-based recommender systems.

**Content-Based Filtering**

We propose to combine TF-IDF (Term Frequency-Inverse Document Frequency) with cosine similarity to create a profile for each game by analyzing its descriptive text. It treats the combined information from the tags, genres, and developer fields as a document. TF-IDF converts this text into a numerical vector, giving more weight to tags that are specific and descriptive. Cosine similarity is then used to calculate the "angle" between these vectors, providing a score of how similar two games are in content.

It's computationally efficient, easy to interpret and doesn't require any user data, making it a perfect starting point.

**Collaborative Filtering**

This is the core of personalization, leveraging the interaction data from user-item and user reviews datasets to model user taste.

Matrix factorization (specifically, Alternating Least Squares - ALS[^9]) is a technique that decomposes the massive user-item interaction matrix into two smaller, denser matrices: a "user-factor" matrix and an "item-factor" matrix. These factors are latent (hidden) features, like "affinity for open-world games" or "preference for high-difficulty." The model learns these factors by trying to reconstruct the original interaction data. ALS is the industry standard[^10] for this task because it's highly scalable (designed for distributed systems like Spark) and specifically tailored for implicit feedback. We can feed it the log-transformed `playtime_forever` as the confidence score of a user's preference, which is a perfect fit for the data. This method uncovers unexpected recommendations that a user might not have found otherwise.

**Hybrid Mode**

Instead of using these methods in isolation, we can combine them into a single, powerful system. Here is a proposed method that's taking advantage of a hybrid model:

- **Candidate Generation**: In this stage, we use the faster, broader models (ALS collaborative filtering and content-based similarity) to generate a large pool of several hundred potentially relevant candidates for a user. For a given `user_id`, this model can rapidly find the top games that the user is most likely to enjoy based on the latent factors learned from the community's playtime data. As a consequence, we will get an output consisting of a list of personalized game IDs. The system then looks at the last few games the user has played or positively reviewed. For each of these seed games, it uses the pre-calculated TF-IDF vectors to find the most similar games based on tags and genres.

- **Fine-Grained Ranking**: Now use a more complex machine learning model, like gradient boosted trees (e.g., XGBoost, LightGBM), to rank this smaller set of candidates. This ranking model takes the outputs of all the other methods as its input features. Gradient boosted trees are excellent at handling a mix of numerical and categorical features and are renowned for their high performance in ranking tasks[^11]. For each of the ~300 candidate games, the system creates a rich feature vector. The ranking model then takes this feature vector and outputs a single score representing the probability that the user will interact with that specific item.

Finally, the system sorts the candidates by this new score in descending order, and the top 10~20 are displayed to the user.

### 3.2 Justification

**Pure Content-Based System**

A content-based system is good at **solving cold start problem**. A brand new game can be recommended the moment it's added to the catalog, as long as it has metadata. The system doesn't need to wait for users to play or review it.

It doesn't require any data from other users to make a recommendation for a specific user. This makes the underlying **calculations simpler** and avoids some **privacy considerations**.

However, if a user only plays say open-world RPGs, this system will only ever recommend other open-world RPGs. It will never make the leap to suggest a highly-rated deck-builder game that the user might unexpectedly love, **limiting the scope of recommendation**.

**Pure Collaborative Filtering System**

By analyzing patterns across thousands of users, it can discover that players who enjoy complex strategy games also tend to enjoy intricate puzzle games, even if the genres seem unrelated. This leads to **novel and delightful discoveries**.

Beside, the model learns directly from user behavior, which is often a **more reliable** indicator of preference than curated metadata.

On the other hand, it's useless for new items that have no interaction data and cannot generate personalized recommendations for new users who have no playtime history. And these models have a natural tendency to recommend items that are already popular, creating a **feedback loop** that makes popular items even more popular.

**Hybrid System**

Relying on a single recommendation method creates vulnerabilities. For example, a purely collaborative filtering system might fail for new users and new games (cold start) and can struggle to recommend items outside a user's established preferences. In contrary, a solely content-based system can trap users in a filter bubble, only recommending items that are almost identical to what they've already played and missing novel discoveries. A hybrid system's principle is to combine the signals from all models to create a final recommendation that is greater than the sum of its parts.

The main drawback is that the system is more **complex to design**, build, and maintain. However, this is a necessary trade-off for the significant gains in performance and user experience.





[^9]: [A gentle introduction to Alternating Least Squares](https://sophwats.github.io/2018-04-05-gentle-als.html)

[^10]: [Tutorial: Create, evaluate, and score a recommendation system](https://learn.microsoft.com/en-us/fabric/data-science/retail-recommend-model)

[^11]: [Gradient Boosted Decision Trees](https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt)


## 4 Evaluation

- [ ] suitable metrics
  - [ ] for model
  - [ ] for system
- [ ] identify the most important metric
- [ ] tradoffs between different metrics
- [ ] consideration for choosing the best model/system
- [ ] computational requirements
  - [ ] real-life practicability
  - [ ] dynamically updated
- [ ] user study
  - [ ] real users
  - [ ] simulated UI
  - [ ] feedback (e.g. questionaire)

### 4.1 Evaluating the Recommendation Model

The standard approach is to create a training/test split. For each user, we hide their most recent interactions and train the model on their remaining history. The goal is to see how well the model can predict these hidden items.

The following metrics evaluate how good the model is at placing relevant items at the top of the recommendation list.

**Precision@N**

This measures the percentage a user actually play/positively review in the test set among the top K games recommended. With precision@N, we can determine how many of the recommendations were actually relevant.

**Recall@N**

This measures the proportion of successfully recommend in the top N list? This reflects to how many did we find of all the games the user liked.

Use Case: These are great for evaluating carousels with a fixed number of slots, like a "Top 10 Picks for You."

**Mean Average Precision (MAP@N)**

Mean average precision is a more robust metric than precision@N because it heavily rewards a model for placing relevant items at the very top of the list. A correct recommendation at rank #1 is scored much higher than one at rank #10.

It's Ideal for measuring the performance of primary, ordered recommendation lists where the top few results are most critical.

**Normalized Discounted Cumulative Gain (NDCG@N)**

NDCG compares rankings to an ideal order where all relevant items are at the top of the list. NDCG@N is determined by dividing the discounted cumulative gain (DCG) by the ideal DCG representing a perfect ranking [^12]. Its key advantage is that it can handle graded relevance scores. Instead of a simple relevant/not relevant flag, we can use the log-transformed playtime_forever as the relevance score.

NDCG would correctly identify that recommending a game a user played for 200 hours is far better than recommending one they played for 20 minutes. It rewards the model for predicting not just what users will like, but how much they will like it.

### 4.2 Beyond Accuracy

A system that is accurate but only recommends obvious bestsellers is not a good system. The following metrics also evaluate the quality of the recommendations themselves.

**Coverage**

Coverage is calculated as the percentage of the total items in the catalog that the model recommend over a large sample of users. It helps diagnose if our model is suffering from popularity bias. Low coverage means it's consistently failing to recommend items from the set.

**Diversity**

Diversity measures how different the recommended items are from each other. This can be done by calculating the average content distance (using the TF-IDF vectors) between items in a recommendation list. A high average distance means high diversity. This ensures we are not creating a filter bubble. It's the key metric for evaluating whether the system is helping users discover new interests.

### 4.3 Best Practice with NDCG@N

For this specific recommendation problem, we've decided the single most important offline metric is NDCG@K.

The superiority of NDCG for this project comes down to that it embraces graded relevance. The datasets contain more feedback data than only a binary like/dislike feature (like playtime). NDCG allows us to use value like `log(playtime_forever)` as the relevance score. This means correctly recommending the 500-hour game contributes significantly more to the overall performance score than recommending the 2-hour game. This makes NDCG a much more sensitive and realistic measure of a model's ability to predict true user satisfaction.

**Models Evaluation**

First we should determine a fixed `N` for NDCG@N as the main target. This should maximize the score on our test set. Then train all candidate models on the training data. Evaluate each one on the test set, calculating NDCG@N, coverage, and diversity for each.

Discard any models that fall below our minimum threshold for coverage. A model that isn't capable of recommending from the long-tail is a non-starter, regardless of its accuracy.

From the remaining pool of models, we will select the one with the highest NDCG@N score. This is the one that is best at ranking highly relevant items at the top of the list.

### 4.4 Computational Resource Consideration

The offline stage (including Training the collaborative filtering and content-based model) is characterized by high-throughput, batch workloads. It demands a powerful, scalable data processing cluster (e.g., Spark on AWS EMR or Google Dataproc) but has relaxed latency requirements. This is the periodic, batch-processing workload where the models are trained on the full datasets. This phase is computationally expensive, but it does not need to be real-time. It can be run daily or weekly on a schedule.

The online stage (including candidate generation and ranking) is characterized by low-latency, high-concurrency requests. It relies heavily on pre-computation from the offline stage and uses specialized, in-memory databases and optimized serving infrastructure to deliver recommendations in near-real time. This is the workload that happens a user visits the service and needs recommendations instantly. The primary requirement here is extremely low latency (typically under 200 milliseconds for the entire process).

### 4.5 User Study

To determine if our system helps users discover games they are genuinely interested in, we have to create a mock-up interface to get the real-world feedback

Imagine we have a pool of test users. For each user, use their existing data from user-ietm dataset to generate a set of personalized recommendations using our model. As a baseline, also generate a list of the globally most popular games.

`Questionnaire`

Now present a user with a list of 10-20 recommendations from our model.

- Task 1 (Relevance): Ask them to go through the list and mark each game with "Interested," "Not Interested," or "Already Played."
- Task 2 (Discovery): Ask them to identify if there are any games on the list that they have never heard of but are now interested in trying.
- Task 3 (Comparison): Show them the baseline "most popular" list and ask them to perform the same tasks.

**Collecting Feedback**

For collecting quantitative feedback on the survey ratings:

- Perceived Relevance: (1-5 scale) "The recommendations were relevant to my tastes."
- Perceived Novelty: (1-5 scale) "I discovered new and interesting games."
- Overall Satisfaction: (1-5 scale) "Overall, I was satisfied with these recommendations."

[^12]: [Normalized Discounted Cumulative Gain (NDCG) explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)