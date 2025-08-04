import pandas as pd
import os
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Define the files and their corresponding method names ---
# This makes it easy to add or remove files later.
files_to_merge = {
    'tfidf': 'tfidf_ndcgs.csv',
    'llm': 'llm_ndcgs.csv',
    'svm': 'svm_ndcgs.csv',
    'vn': 'vn_ndcgs.csv',
    'nn': 'nn_ndcgs.csv',
    # 'nopain': 'nopain_ndcgs.csv' # Assuming 'nopain' is the method name for this file
}

# --- 2. Load and prepare each file ---
# We'll store each prepared DataFrame in a list.
data_frames = []

print("--- Reading and preparing individual result files ---")
for method_name, file_name in files_to_merge.items():
    if not os.path.exists(file_name):
        print(f"Warning: File '{file_name}' not found. Skipping.")
        continue
    
    # Read the CSV
    df = pd.read_csv(file_name)
    
    # --- FIX: Make column identification more robust ---
    # Instead of searching for 'NDCG', we find the column that is NOT 'USERNAME'.
    # This avoids issues with inconsistent naming (e.g., 'nDCG@200', 'ndcg', 'score').
    # The .upper() makes the check case-insensitive.
    try:
        score_col_name = [col for col in df.columns if col.upper() != 'USERNAME'][0]
    except IndexError:
        print(f"Warning: Could not find a score column in '{file_name}'. Skipping.")
        continue
        
    # Rename the score column to the specified format
    new_col_name = f'NDCG@200_{method_name}'
    df.rename(columns={score_col_name: new_col_name}, inplace=True)
    
    # Set USERNAME as the index to prepare for merging
    # We also ensure the USERNAME column itself is consistently named.
    username_col = [col for col in df.columns if col.upper() == 'USERNAME'][0]
    df.rename(columns={username_col: 'USERNAME'}, inplace=True)
    df.set_index('USERNAME', inplace=True)
    
    data_frames.append(df)
    print(f"Processed '{file_name}' for method '{method_name}'.")

# --- 3. Merge all DataFrames together ---
# We use reduce to sequentially merge all dataframes in the list on their index (USERNAME).
if not data_frames:
    print("No data files were processed. Exiting.")
    exit()

print("\n--- Merging all method results ---")
combined_df = reduce(lambda left, right: pd.merge(left, right, on='USERNAME', how='outer'), data_frames)

# --- 4. Load and merge vote counts ---
print("--- Adding user vote counts ---")
vote_counts_file = 'vote_count_top20.csv'
if os.path.exists(vote_counts_file):
    vote_counts_df = pd.read_csv(vote_counts_file)
    # Standardize column names for merging
    vote_counts_df.rename(columns={'count': 'vote_count', 'USERNAME': 'USERNAME'}, inplace=True)
    vote_counts_df.set_index('USERNAME', inplace=True)
    
    # Merge vote counts with the combined results
    final_df = pd.merge(vote_counts_df, combined_df, on='USERNAME', how='left')
else:
    print(f"Warning: '{vote_counts_file}' not found. Vote counts will not be added.")
    final_df = combined_df # Proceed without vote counts if file is missing

# --- 5. Sort the results ---
# We sort by 'vote_count' in descending order.
if 'vote_count' in final_df.columns:
    print("--- Sorting results by vote count ---")
    final_df = final_df.sort_values(by='vote_count', ascending=False)
# --- 6. Round the NDCG scores ---
print("--- Rounding NDCG scores to 3 decimal places ---")
# Find all columns that contain the NDCG scores
ndcg_cols = [col for col in final_df.columns if 'NDCG@200' in col]
# Round each of these columns to 3 decimal places
for col in ndcg_cols:
    final_df[col] = final_df[col].round(4)
# --- 6. Save the final CSV ---
output_filename = 'combined_ndcg_results.csv'
final_df.to_csv(output_filename)

print(f"\nSuccessfully merged all results into '{output_filename}'.")
print("\nFinal DataFrame preview:")
print(final_df.head())


# --- 8. Plot the Comparison Chart ---
print("\n--- Generating Method Comparison Chart ---")

def plot_method_comparison(results_df):
    """Generates a grouped line chart comparing NDCG scores for all methods."""
    
    # Prepare data for plotting
    plot_data = results_df.copy()
    if 'vote_count' in plot_data.columns:
        plot_data['user_label'] = plot_data.index + '/' + plot_data['vote_count'].astype(str)
        plot_data.set_index('user_label', inplace=True)
    
    ndcg_columns = [col for col in plot_data.columns if 'NDCG@200' in col]
    plot_data_ndcg = plot_data[ndcg_columns]
    
    # Define different line styles and lighter weight
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 12))
    
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))] # Solid, dashed, dash-dot, dotted, custom dash
    line_weight = 2.2 # Reduced line weight

    for i, column in enumerate(plot_data_ndcg.columns):
        plot_data_ndcg[column].plot(
            kind='line', 
            ax=ax,
            marker='o', 
            markersize=7, 
            linestyle=styles[i % len(styles)], # Cycle through styles
            lw=line_weight,
            label=column.replace('NDCG@200_', '') # Clean label for legend
        )

    ax.set_title('Per-User NDCG@200 Comparison Across All Methods', fontsize=20, pad=20)
    ax.set_ylabel('NDCG@200 Score', fontsize=16)
    ax.set_xlabel('User / Total Votes', fontsize=16)
    
    # --- FIX: Ensure all x-tick labels are displayed ---
    # Set ticks for every user to prevent matplotlib from skipping any.
    ax.set_xticks(np.arange(len(plot_data_ndcg.index)))
    # Set the labels for those ticks.
    ax.set_xticklabels(plot_data_ndcg.index, rotation=45, ha='right', fontsize=12)
    
    # Customize legend
    ax.legend(title='Method', fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    
    # Set y-axis limits to give some space
    ax.set_ylim(bottom=-0.05, top=plot_data_ndcg.max().max() * 1.1)
    
    plt.tight_layout()
    plt.savefig("cmp.png")
    plt.show()

# Call the plotting function with the final DataFrame
plot_method_comparison(final_df)



# --- 9. Calculate and Display Mean NDCG Scores ---
print("\n--- Calculating Mean NDCG@200 Scores ---")

# We use the final_df which has the rounded scores
ndcg_columns = [col for col in final_df.columns if 'NDCG@200' in col]
mean_scores = final_df[ndcg_columns].mean()

# Rename the index for better readability
mean_scores.index = [idx.replace('NDCG@200_', '') for idx in mean_scores.index]

# Sort the results for a clear ranking
mean_scores = mean_scores.sort_values(ascending=False)

print("\nAverage NDCG@200 Score per Method:")
print(mean_scores.to_string())
