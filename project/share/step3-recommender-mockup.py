'''
This recommender is still in development.
Critical issue happens when doing personalized recommendations
because  the user history mechanisim is faulty.
'''


import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# 1. data loading and preparation ---
# This section should be run only once at the start of the application.
print("--- Initializing Recommender System ---")
print("Loading data and embeddings... (This may take a moment)")

# load embeddings
embeddings_data = np.load('showerthoughts_combined_embeddings.npz')
# title_embeddings = embeddings_data['title_embed']
# content_embeddings = embeddings_data['content_embed']
# combined_llm_embeddings = (0.6 * title_embeddings) + (0.4 * content_embeddings)
# combined_llm_embeddings = embeddings_data['combined_embed']

# Load post data
posts_df = pd.read_csv('reddit_showerthoughts.tsv', sep='\t', on_bad_lines='skip').dropna(subset=['title'])
posts_df = posts_df.reset_index(drop=True)
posts_df['embedding_idx'] = posts_df.index


print("System ready.")
print("-" * 40)


# 2. core recommender logic ---

class Recommender:
    def __init__(self, posts_data, embeddings_data):
        self.posts_df = posts_data
        self.embeddings = embeddings_data['combined_embed']
        self.user_history = [] # list to store user votes as dicts
        # add a set to track all seen posts (voted or skipped)
        self.seen_ids = set()
        self.cold_start_threshold = 10

    def get_trending_posts(self, n=5):
        """Returns top N posts based on a simple score, excluding seen posts."""
        # rank by trend
        trending_score = (np.log(self.posts_df['ups'].fillna(0).add(1)) / np.log(50)) +self.posts_df['upvote_ratio'].fillna(0)
        print(trending_score.head(5))
        # get all trending indices and filter out seen ones
        posts_df['trending']=trending_score
        all_trending_ids = trending_score.argsort()[::-1]
        posts_by_trending_df=posts_df.sort_values(by='trending',ascending=False)
        # print(posts_by_trending_df.head(10))
        unseen_trending_ids = [id for id in posts_by_trending_df['submission_id'] if id not in self.seen_ids]
        
        # Return the top N of the unseen posts
        return self.posts_df[posts_df['submission_id'].isin(unseen_trending_ids[:n])]

    def get_personalized_recommendations(self, n=5):
        """Generates personalized recommendations based on user history."""
        history_df = pd.DataFrame(self.user_history)
        
        upvoted_indices = history_df[history_df['vote'] == 'upvote']['embedding_idx'].tolist()
        downvoted_indices = history_df[history_df['vote'] == 'downvote']['embedding_idx'].tolist()

        # handle cases where user has only upvoted or downvoted
        if not upvoted_indices or not downvoted_indices:
            print("\n[INFO] Need both upvotes and downvotes for personalized predictions.")
            print("[INFO] Falling back to recommendations based on your upvotes...")
            if not upvoted_indices: return self.get_trending_posts(n) # Fallback to trending
            
            profile = self.embeddings[upvoted_indices].mean(axis=0).reshape(1, -1)
            sim = cosine_similarity(profile, self.embeddings).flatten()
            
            # exclude all seen posts
            sim[list(self.seen_ids)] = -1
            
            rec_indices = sim.argsort()[::-1][:n]
            return self.posts_df.iloc[rec_indices]


        # use the vote prediction model logic
        upvote_profile = self.embeddings[upvoted_indices].mean(axis=0).reshape(1, -1)
        downvote_profile = self.embeddings[downvoted_indices].mean(axis=0).reshape(1, -1)

        # create features for all posts the user hasn't seen
        unseen_ids = self.posts_df.index.difference(list(self.seen_ids)).tolist()
        unseen_vectors = self.embeddings[unseen_ids]

        sim_to_upvotes = cosine_similarity(unseen_vectors, upvote_profile).flatten()
        sim_to_downvotes = cosine_similarity(unseen_vectors, downvote_profile).flatten()
        
        X_predict = np.vstack([sim_to_upvotes, sim_to_downvotes]).T

        # Ttrain a classifier on the user's history
        X_train_indices = history_df['embedding_idx'].tolist()
        X_train = np.vstack([
            cosine_similarity(self.embeddings[X_train_indices], upvote_profile).flatten(),
            cosine_similarity(self.embeddings[X_train_indices], downvote_profile).flatten()
        ]).T
        y_train = (history_df['vote'] == 'upvote').astype(int)
        
        classifier = LogisticRegression(random_state=42, class_weight='balanced')
        classifier.fit(X_train, y_train)

        # predict probability of upvote for all unseen posts
        upvote_probs = classifier.predict_proba(X_predict)[:, 1]
        
        # get the indices of the top N posts with highest upvote probability
        top_n_local_indices = upvote_probs.argsort()[::-1][:n]
        
        # map local indices back to original dataframe indices
        recommendation_indices = [unseen_ids[i] for i in top_n_local_indices]
        
        return self.posts_df.iloc[recommendation_indices]

    def add_vote(self, post_index, vote):
        """Adds a user's vote to their history."""
        post = self.posts_df.iloc[post_index]
        self.user_history.append({
            'embedding_idx': post['embedding_idx'],
            'title': post['title'],
            'vote': vote
        })
        print(f"\nVote recorded: You '{vote}d' -> '{post['title']}'")

    def run(self):
        """Main interaction loop for the recommender."""
        print("\nWelcome to the ShowerThoughts Recommender!")
        print("Type 'u' to upvote, 'd' to downvote, 's' to skip, or 'q' to quit.")
        
        while True:
            if len(self.user_history) < self.cold_start_threshold:
                print(f"\n--- Phase 1: Cold Start (Top Trending Posts) ---")
                print(f"({len(self.user_history)}/{self.cold_start_threshold} votes recorded)")
                posts_to_show = self.get_trending_posts()
            else:
                print("\n--- Phase 2: Personalized Recommendations ---")
                posts_to_show = self.get_personalized_recommendations()

            if posts_to_show.empty:
                print("\nNo more posts to recommend! Thanks for using the system.")
                break

            for index, post in posts_to_show.iterrows():
                print("\n" + "="*40)
                print(f"Post: {post['title']}")
                
                # mark post as seen as soon as it's displayed
                self.seen_ids.add(post['submission_id'])
                
                action = input("Your action (u/d/s/q): ").lower()
                
                if action == 'q':
                    print("\nThanks for using the recommender. Goodbye!")
                    return
                elif action in ['u', 'd']:
                    vote = 'upvote' if action == 'u' else 'downvote'
                    self.add_vote(index, vote)
                    break 
                elif action == 's':
                    print("[Skipped]")
                    continue
                else:
                    print("[Invalid input. Skipping post.]")
            else: 
                print("\nFinished list. Generating new recommendations...")


# 3. start the app
if __name__ == '__main__':
    recommender_app = Recommender(posts_df, embeddings_data)
    recommender_app.run()

