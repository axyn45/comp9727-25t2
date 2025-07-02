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

Ideally we're going to build a Web UI that allows gamers to interact with the system[^1]. Gamers will be presented with 10-20 games as tiles on the webpage, each with basic information displayed like game title, category/tags and overall rating (if applicable). By clicking on the tile, user can see the full description and some reviews (if applicable) of the game.

[^1]: This is not guranteed and maybe replaced by a text-based simulated solution.

### 1.4 Simulating User Inputs

With the web-based interface mentioned before, we can capture behavioral traits from the user to refine our recommendation.

While browsing games, a user may be given the basic information, such as the name, developer, price and category of a title. If a user would like to know more on a specific game, they may click the title and jump to the profile page of the game. The clicking action indicates that the user showed interest in this game, meaning it would be sensible to make more precise and personalized reommendation based on the current game title. This action will be logged and sent back to the backend to update/improve the model.

Besides browsing and clicking, a user may also add games into wishlist if the title is appealing. This will give the wishlisted games a high weight for future recommendation, meaning games similar to the ones in the wishlist have a higher chance to be faved by the player.

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

- [ ] describe the dataset
- [ ] sufficient quality and quantity
- [ ] which field is helpful in which way
- [ ] limited breadth
- [ ] unrealistic data
- [ ] overfit issue

**Steam Video Game and Bundle Data**



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
