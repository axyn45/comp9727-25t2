# Project: Gamer's Partner

In an era of abundant digital entertainment, players often face the paradox of choice, struggling to find new games that align with their unique tastes. The sheer volume of available titles on platforms like Steam, Epic Games Store, and console marketplaces makes manual discovery a significant challenge. This document details the design of a sophisticated recommender system aimed at solving this problem. By intelligently analyzing player behavior, preferences, and the inherent attributes of games, our system will provide tailored recommendations that connect players with their next favorite game. The core objectives are to increase player satisfaction, drive engagement, and create a more personalized and compelling user experience.

## 1 Scope

- [x] what's the domain and target user
- [x] how many items presented and through what way
- [x] simulate user input
- [ ] address updating model and cold start
- [ ] business consideration

### 1.1 Project Domain

This project operates within the video game and digital entertainment industry. The system is designed to function within a digital storefront or platform where games are sold and distributed (e.g., Steam, Epic Games Store, GOG, or a console marketplace like the PlayStation Store). The primary role is to drive user activity and purchases within this commercial environment.

Beyond profit consideration, the system also focus on enhancing the user experience. By making discovery easier and more personal, it aims to keep players invested in the platform, encouraging them to return, play more, and explore a wider variety of the catalog.

### 1.2 Target User

This system targets the players on game distribution platforms. Essentially we can categorize our intended users into these two groups:

1. **Newcomer** who is new to the platform or to gaming in general. They are often overwhelmed by a large catalog and need clear, accessible recommendations.

2. **Explorer** who enjoys discovering unique, niche, or indie titles that fall outside of the mainstream best-sellers. They value recommendations that are surprising and align with their specific, often eclectic, tastes. The recommender system helps them sift through thousands of titles to find these games.

### 1.3 Delivery of Recommendation

Ideally we're going to build a Web UI that allows gamers to interact with the system[^1] . Gamers will be presented with 10-20 games as tiles on the webpage, each with basic information displayed like game title, category/tags and overall rating (if applicable). By clicking on the tile, user can see the full description and some reviews (if applicable) of the game. 

[^1]: this is not guranteed and maybe replaced by a simulated solution

### 1.4 Simulating User Inputs

With the web-based interface mentioned before, we can capture behavioral traits from the user to refine our recommendation.

While browsing games, a user may be given the basic information, such as the name, developer, price and category of a title. If a user would like to know more on a specific game, they may click the title and jump to the profile page of the game. The clicking action indicates that the user showed interest in this game, meaning it would be sensible to make more precise and personalized reommendation based on the current game title. This action will be logged and sent back to the backend to update/improve the model.

Besides browsing and clicking, a user may also add games into wishlist if the title is appealing. This will give the wishlisted games a high weight for future recommendation, meaning games similar to the ones in the wishlist have a higher chance to be faved by the player.

## 2 Datasets
- [ ] sufficient quality and quantity
- [ ] which field is helpful in which way
- [ ] limited breadth
- [ ] unrealistic data
- [ ] overfit issue

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


