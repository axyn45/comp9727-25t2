import kagglehub
import math
import os
import pandas as pd
import praw
import re
from tqdm.notebook import tqdm
import string

s = "some\x00string. with\x15 funny characters"
printable = set(string.printable)
print(''.join(filter(lambda x: x in printable, s)))

def download_dataset()->list[str]:
    """
    Download the dataset from Kaggle and return the paths to the files.
    """
    dataset_dir = kagglehub.dataset_download("josephleake/huge-collection-of-reddit-votes")
    paths = []
    for dir_path, _, file_names in os.walk(dataset_dir):
        for file_name in file_names:
            paths.append(os.path.join(dir_path, file_name))
    print(f'File path to votes:\n{paths[0]}')
    print(f'File path to submissions:\n{paths[1]}')
    return paths

def get_dataframe()->tuple[pd.DataFrame]:
    """
    Return a tuple of two pandas.Dataframe: votes and submissions.

    Returns:
        tuple[pd.DataFrame]: a tuple of two dataframes.
    """
    paths = download_dataset()
    votes = pd.read_csv(paths[0], sep='\t')
    submissions = pd.read_csv(paths[1], sep='\t')
    return (votes, submissions)

def view_users_votes(votes:pd.DataFrame):
    view = (
        votes
        .groupby(['USERNAME', 'SUBREDDIT', 'VOTE'])
        .size()                         # count upvotes/downvotes in each group
        .unstack(fill_value=0)          # pivot VOTE labels into columns
        .rename(columns={
            'upvote':   'num_upvotes',
            'downvote': 'num_downvotes'
        })
        .reset_index()                  # turn USERNAME & SUBREDDIT back into columns
    )
    return view

def filter_subreddits(
    votes: pd.DataFrame,
    num_upvotes: int = 0,
    num_downvotes: int = 0,
    total_votes: int = 0,
    num_users: int = 0,
) -> pd.DataFrame:
    """
    Filter a DataFrame of subreddit vote counts according to given thresholds.

    Parameters:
    - votes: DataFrame with at least ['USERNAME', 'SUBREDDIT', 'num_upvotes', 'num_downvotes'] columns.
    - num_upvotes: keep rows where num_upvotes > this value (if > 0).
    - num_downvotes: keep rows where num_downvotes > this value (if > 0).
    - total_votes: keep rows where (num_upvotes + num_downvotes) > this value (if > 0).
    - num_users: keep rows where the subreddit has more than this many unique users (if > 0).

    Returns:
    - Filtered DataFrame.
    """
    # Start with an all-True mask
    mask = pd.Series(True, index=votes.index)

    # Apply upvotes threshold
    if num_upvotes > 0:
        mask &= votes['num_upvotes'] > num_upvotes

    # Apply downvotes threshold
    if num_downvotes > 0:
        mask &= votes['num_downvotes'] > num_downvotes

    # Apply total votes threshold
    if total_votes > 0:
        mask &= (votes['num_upvotes'] + votes['num_downvotes']) > total_votes

    # Apply distinct user count per subreddit threshold
    if num_users > 0:
        # Compute number of unique users for each subreddit
        user_counts = votes.groupby('SUBREDDIT')['USERNAME'].transform('nunique')
        mask &= user_counts > num_users

    # Return a series of subreddits
    return pd.Series(votes[mask].copy()['SUBREDDIT'].unique()).str.replace(r'^r/', '', regex=True)

def get_post_count(
        subreddits:pd.Series,
        submissions:pd.DataFrame
) -> pd.DataFrame:
    """Tally the number of posts for each subreddit provided in the parameters.

    Args:
        subreddits (pd.Series): a list of names of subreddits
        submissions (pd.DataFrame): a batch of submissions/posts

    Returns:
        pd.DataFrame: a dataframe that contains the name of subreddits and number of posts in them.
    """
    posts = (
        submissions
            .groupby(['SUBREDDIT'])
            .size()
            .reset_index(name='POSTS')
            .sort_values('POSTS', ascending=False)
    )
    return posts[posts['SUBREDDIT'].isin(subreddits)]

def download_posts(
    submissions: pd.DataFrame,
    reddit_instance: praw.Reddit,
    savefile_path: str="reddit_data.csv"
):
    """Download metadata of posts from provided submissions.

    Args:
        submissions (pd.DataFrame): A DataFrame with columns "index" and "SUBMISSION_ID".
        reddit_instance (praw.Reddit): Your reddit API instance
        savefile_path (str, optional): File to which meta data is written. Defaults to "reddit_data.csv".
    """
    def make_post_entry(
        index=0,
        submission_id=0,
        title="",
        selftext="",
        num_comments=0,
        num_unique_commentators=0,
        ups=0,
        upvote_ratio=0,
        author="",
        created_utc=0,
        text_only=True,
    ):
        """
        Make data entry from post for building a DataFrame
        """
        return {
            "index": index,
            "submission_id": submission_id,
            "title": title,
            "selftext": selftext.replace('\n',' ').replace('\t',' '),
            "num_comments": num_comments,
            "num_unique_commentators": num_unique_commentators,
            "ups": ups,
            "upvote_ratio": upvote_ratio,
            "author": author,
            'created_utc': created_utc,
            'text_only': text_only,
        }
    
    def entry_to_text(entry):
        utf_8_string= str(entry["index"])+'\t'+str(entry["submission_id"])+'\t'+str(entry["title"])+'\t'+str(entry["selftext"])+'\t'+str(entry["num_comments"])+'\t'+str(entry["num_unique_commentators"])+'\t'+str(entry["ups"])+'\t'+str(entry["upvote_ratio"])+'\t'+str(entry["author"])+'\t'+str(entry["created_utc"])+'\t'+str(entry["text_only"])
        return ''.join(filter(lambda x: x in printable, utf_8_string))

    # Define the schema we expect
    expected_cols = list(make_post_entry())

    try:
        # If the file exists, verify its header matches exactly
        if os.path.isfile(savefile_path):
            existing_cols = pd.read_csv(savefile_path, nrows=0).columns.tolist()
            if existing_cols != expected_cols:
                print(
                    f"Header mismatch:\n"
                    f"  existing file has columns {existing_cols}\n"
                    f"  expected columns {expected_cols}"
                )
                return None
        else:
            pd.DataFrame(columns=expected_cols).to_csv(savefile_path, index=False)
    except e:
        print(f'Invalid path {savefile_path}: {e}')

    buffer = []
    id_pattern = r"^t3_" # used for sanitizing id string
    chunk_size=100
    chunks = math.ceil(submissions.shape[0] / chunk_size)
    counter = 0
    f_reddit = open("reddit_data.tsv","w", encoding="utf-8")

    for i in tqdm(range(chunks), "working hard at scraping", chunks):
        start = i*chunk_size
        end = (i+1)*chunk_size
        slc = slice(start, end)
        try:
            posts = reddit_instance.info(fullnames=submissions.iloc[slc]['SUBMISSION_ID'].to_list())
            for idx, post in enumerate(posts):
                entry = make_post_entry(
                    index = idx + start,
                    submission_id = post.name,
                    title = post.title,
                    selftext = post.selftext,
                    num_comments = len(post._comments_by_id),
                    num_unique_commentators = len(set(post._comments_by_id.keys())),
                    ups = post.ups,
                    upvote_ratio = post.upvote_ratio,
                    author = post.author.name if post.author is not None else "",
                    created_utc = post.created_utc,
                    text_only = (not post.is_video) and (post.media is None),
                )
                f_reddit.write(entry_to_text(entry)+'\n')
                
                # buffer.append(entry)

            # df = pd.DataFrame(buffer, columns=expected_cols)
            # df.to_csv(savefile_path, mode='a', header=False, index=False)
            # counter += len(buffer)
            # buffer.clear()
        except Exception as e:
            print(f"Error processing batch {start}-{end}: {e}")
    f_reddit.close()
    print(f"Downloaded {counter} posts from {len(submissions)} ids.")

reddit = praw.Reddit(
    "scrapper",
    user_agent="rs_scrapper",
)

votes, submissions = get_dataframe()
showerthoughts = submissions[submissions['SUBREDDIT'].isin(['Showerthoughts'])].reset_index(drop=True)

download_posts(showerthoughts, reddit)

showerthoughts_data = pd.read_csv('reddit_data.csv')