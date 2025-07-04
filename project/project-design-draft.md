# Project: Gamer's Partner

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

We will create a detailed "profile" or "feature vector" for every game using its metadata. This allows us to calculate a similarity score between any two games. For this, we will utlizing the following fields in the dataset:

- `tags`: This is the most important field. It contains highly specific, user-generated descriptors like 'Open World', 'Crafting', 'Roguelike' and 'Atmospheric'. We would process this list of tags for each game using techniques like TF-IDF, to create a numerical vector that represents its unique characteristics.
- `genres`: This provides broader categories like 'Action', 'RPG', or 'Strategy'. This is a high-level feature that helps match games in the same category.
- `specs`: This field describes the game's features, such as 'Single-player', 'Multi-player', or 'Steam Achievements'. It adds another layer of detail for finding similar games. For example, a user who plays many 'Single-player' games can be recommended others with the same spec.
- `developer`: The developer is a very strong signal. Players who enjoy one game from a specific developer (e.g., 'CD PROJEKT RED' or 'Supergiant Games') are highly likely to enjoy their other titles, because games coming from same developer are likely to share similar genres and specs. We can treat this as a high-weight categorical feature.

**Quality Score from User Sentiment**

A game can be very similar in content to another but be of much lower quality. The sentiment field is the key to solving this. It provides aggregated community review scores like 'Overwhelmingly Positive', 'Mostly Positive', or 'Mixed'.

We would convert these categorical labels into a numerical score according to the design of Steam's review system[^6]. For example:

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

This allows the recommender to prioritize games that are not only similar in content but are also well-regarded by the community.

**Generating Recommendations**

When a user expresses interest in a specific game (e.g., they played it, reviewed it, or are currently viewing its page), we use that game as our seed.

The system retrieves the feature vector for the seed game. It then compares this vector to all other games in the dataset and calculates a similarity score (e.g., using cosine similarity). This produces a ranked list of the most similar games based purely on content.

The system then takes this list of similar games and re-ranks it by multiplying the similarity score with the game's quality score (derived from sentiment).

**Filtering**

Other fields in the dataset are essential for giving users control over their recommendations. For example, the recommender can include filters like 'Free to Play', 'Under $10', or '50% OFF' (by comparing `price` and `discount_price`).

Also users can filter to see 'New Releases' or games from a specific era.

And `early_access` is a simple boolean filter to let users decide if they want to see games that are still in development.

With these filters, the system can be more efficient helping users to identify the games that interest them most.

[^6]: [Steam Review System](https://steamcommunity.com/discussions/forum/0/1744482869761322402/#c1744482869761428892)

### 2.3 Limitations

<!-- While the UCSD Steam dataset is fantastic for research and prototyping, using it to model a real-world system reveals several constraints. -->

**Incompleteness**

The dataset is a snapshot, not the whole picture. It dosen't include fine-grained user interaction data, whcih is the most crucial data for a production-level recommender system. Like for example, the dataset is item-centric. There are no user profiles, individual user ratings or explicit links between users and the games they've interacted with. This makes it impossible to build a true collaborative filtering model, which is the cornerstone of most modern recommenders. We have to rely on fields like the aggregated sentiment, or maybe external datasets that include UGC data.

<!-- **Unrealisticity** -->

**Overfitting**

This dataset was originally structured for a well-defined academic task: given a game's metadata, find or predict things about it. This encourages models that are optimized for a static "item-to-item" similarity task but may not be suitable for a dynamic production environment.

Because the dataset is rich in metadata and poor in user-item interactions, it heavily encourages the development of content-based filtering models. While useful, an over-reliance on this approach can lead to "filter bubbles"[^7] where users are only ever shown things that are very similar to what they already know.

Also,the static nature means the model doesn't have to deal with critical production challenges like the cold-start problem, evolving user tastes over time, or temporal changes in gaming trends.

[^7]: [How algorithms and filter bubbles decide what we see on social media](https://www.bbc.co.uk/bitesize/articles/zd9tt39)


## 3 Methods

- [ ] propose different methods
  - [ ] different types of RS
  - [ ] propose an approach
    - [ ] different methods or combined
- [ ] justify their suitability
  - [ ] evaluate methods and system

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


Simulating User Interactions for Evaluation
Since you are not building the UI, you need to simulate it to evaluate your model.

The User Study Plan:

Goal: To determine if your hybrid recommender helps users discover games they are genuinely interested in, compared to a baseline (e.g., a simple "most popular" list).

Setup:

Create a simple command-line or web interface.

Select a pool of test users. For each user, use their existing data from australian_users_items.json to generate a set of personalized recommendations using your model.

As a baseline, also generate a list of the globally most popular games (top overall playtime).

User Tasks:

Present a user with a list of 10-15 recommendations from your model.

Task 1 (Relevance): Ask them to go through the list and mark each game with "Interested," "Not Interested," or "Already Played."

Task 2 (Discovery): Ask them to identify if there are any games on the list that they have never heard of but are now interested in trying.

Task 3 (Comparison): Show them the baseline "most popular" list and ask them to perform the same tasks.

Collecting Feedback:

Quantitative Metrics:

Precision@K: Of the top K recommendations, what percentage was the user interested in?

Novelty/Serendipity: How many "new and interesting" games did your model find for them compared to the baseline?

Qualitative Feedback (Post-Study Survey):

"On a scale of 1-5, how relevant were the recommendations?"

"Did you feel the recommendations were personalized to your tastes?"

"Did the system help you discover new games you wouldn't have found otherwise?"

"Which list (yours or the baseline) did you find more useful, and why?"