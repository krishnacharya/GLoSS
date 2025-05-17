import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.project_dirs import get_reviews_raw2014_dir, processed_data_dir
import os
import argparse
from pathlib import Path
import urllib.parse

def load_and_preprocess_reviews(file_path):
    """Loads and preprocesses the reviews dataset."""
    reviews_df = pd.read_json(file_path, lines=True)
    cols = ['reviewerID', 'asin', 'unixReviewTime', 'overall']
    reviews_df = reviews_df[cols]
    return reviews_df

def deduplicate_reviews(df):
    """Deduplicates reviews based on reviewerID, timestamp, asin, and overall rating."""
    df_withdup = df.copy()
    df_withdup = df_withdup.sort_values(by=['reviewerID', 'unixReviewTime']).reset_index(drop=True)
    df_dedup = df_withdup.drop_duplicates(subset=['reviewerID', 'unixReviewTime', 'asin', 'overall'])
    percentage_duplicates = (1 - len(df_dedup) / len(df_withdup)) * 100
    print(f"Percentage of duplicates: {percentage_duplicates:.2f}%")
    print(f"Shape before deduplication: {df_withdup.shape}")
    print(f"Shape after deduplication: {df_dedup.shape}")
    return df_withdup, df_dedup

def visualize_reviewer_counts(df, category):
    """Visualizes the distribution of review counts per reviewer."""
    reviewer_counts = df['reviewerID'].value_counts()
    mean = reviewer_counts.mean()
    std = reviewer_counts.std()

    plt.figure(figsize=(10, 6))
    plt.hist(reviewer_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label=f'+1 SD: {mean + std:.2f}')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1, label=f'-1 SD: {mean - std:.2f}')
    plt.axvline(mean + 2 * std, color='orange', linestyle='dashed', linewidth=1, label=f'+2 SD: {mean + 2 * std:.2f}')
    plt.axvline(mean - 2 * std, color='orange', linestyle='dashed', linewidth=1, label=f'-2 SD: {mean - 2 * std:.2f}')

    plt.title(f'Histogram of Review Counts per Reviewer ({category})')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    print(reviewer_counts.describe())

def check_monotonicity(df, column='unixReviewTime', group_by=None):
    """Checks if a column is monotonically increasing globally and within groups."""
    results = {}
    results['global_monotonic'] = df[column].is_monotonic_increasing
    if group_by:
        results['grouped_monotonic'] = df.groupby(group_by, observed=False)[column].apply(lambda x: x.is_monotonic_increasing).all()
    else:
        results['grouped_monotonic'] = None
    return results

def calculate_asin_fraction(df, meta):
    """Calculates the fraction of unique ASINs in df that are present in meta df."""
    unique_asins_df = set(df['asin'])
    unique_asins_meta = set(meta['asin'])
    if not unique_asins_df:
        return 0.0
    fraction_in_meta = len(unique_asins_df.intersection(unique_asins_meta)) / len(unique_asins_df)
    return fraction_in_meta

def merge_with_metadata(df, meta):
    """Merges the input DataFrame with the metadata DataFrame on the 'asin' column."""
    merged_df = df.merge(meta, on='asin', how='inner')
    return merged_df

def calculate_dfmetrics(df):
    """Calculates dataframe metrics."""
    unique_users = df['reviewerID'].nunique()
    unique_items = len(set(df['asin']))
    total_purchases = df.shape[0]
    avgitems_per_user = df['reviewerID'].value_counts().mean()
    if unique_items > 0:
        avgpurchases_per_asin = total_purchases / unique_items
        density = total_purchases / (unique_users * unique_items)
    else:
        avgpurchases_per_asin = 0
        density = 0

    return {
        "unique_users": unique_users,
        "unique_items": unique_items,
        "total_purchases": total_purchases,
        "avg_items_per_user": avgitems_per_user,
        "avg_purchases_per_item": avgpurchases_per_asin,
        "density": density
    }

def filter_reviewers_with_min_reviews(df, min_reviews=3):
    """Filters the DataFrame to include only reviewers with at least a specified number of reviews."""
    reviewer_counts = df['reviewerID'].value_counts()
    reviewers_with_min_reviews = reviewer_counts[reviewer_counts >= min_reviews].index
    filtered_df = df[df['reviewerID'].isin(reviewers_with_min_reviews)]
    return filtered_df

def create_corpus_metadata(meta, asin_list, output_path):
    """Creates a metadata file containing only items present in the provided ASIN list."""
    corpus_items = meta[meta['asin'].isin(asin_list)]
    corpus_items.to_json(output_path, orient='records', lines=True)
    print(f"Corpus metadata saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process Amazon reviews datasets.")
    parser.add_argument("--reviews_url", type=str, required=True, help="The URL of the reviews JSON.gz file to download.")
    parser.add_argument("--category", type=str, required=True, help="The category of the reviews (used for output directory naming and metadata file lookup).")
    args = parser.parse_args()
    reviews_url = args.reviews_url
    category = args.category.lower().capitalize()

    raw_dir2014 = get_reviews_raw2014_dir() # this is a Pathlib object
    parsed_reviews_url = urllib.parse.urlparse(reviews_url)
    reviews_filename_compressed = os.path.basename(parsed_reviews_url.path)
    reviews_filename_uncompressed, ext = os.path.splitext(reviews_filename_compressed)
    if ext == '.gz':
        reviews_filename_uncompressed = reviews_filename_uncompressed
    else:
        reviews_filename_uncompressed = reviews_filename_compressed

    reviews_file_compressed = str(raw_dir2014 / reviews_filename_compressed)
    reviews_file_uncompressed = str(raw_dir2014 / reviews_filename_uncompressed)
    metadata_file = str(processed_data_dir(f"{category.lower()}2014") / 'meta_proc.json')
    output_dir = processed_data_dir(f"{category.lower()}2014")

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download and unzip the reviews file if it doesn't exist
    if not os.path.exists(reviews_file_uncompressed):
        print(f"Downloading reviews from: {reviews_url} to {reviews_file_compressed}...")
        os.system(f"wget '{reviews_url}' -O '{reviews_file_compressed}'")
        if os.path.exists(reviews_file_compressed) and reviews_filename_compressed.endswith('.gz'):
            print(f"Unzipping {reviews_file_compressed} to {reviews_file_uncompressed}...")
            os.system(f"gunzip '{reviews_file_compressed}' -c > '{reviews_file_uncompressed}'")
        elif not reviews_filename_compressed.endswith('.gz'):
            reviews_file_uncompressed = reviews_file_compressed
        elif not os.path.exists(reviews_file_compressed):
            print(f"Error: Could not download reviews file from {reviews_url}.")
            return
    elif reviews_filename_compressed.endswith('.gz') and not os.path.exists(reviews_file_compressed):
        print(f"Warning: Uncompressed reviews file exists at {reviews_file_uncompressed}, but compressed version not found. Assuming uncompressed version is the latest.")

    # Load and preprocess reviews
    reviews_df = load_and_preprocess_reviews(reviews_file_uncompressed)  # Changed beauty_df to reviews_df

    # Deduplicate reviews
    df_withdup, df_dedup = deduplicate_reviews(reviews_df) # Changed beauty_df to reviews_df

    # Visualize reviewer counts
    visualize_reviewer_counts(df_dedup, category)

    # Check monotonicity of review times
    results_withdup = check_monotonicity(df_withdup, group_by='reviewerID')
    print(f"Monotonicity (with duplicates): Global - {results_withdup['global_monotonic']}, Grouped - {results_withdup['grouped_monotonic']}")
    results_dedup = check_monotonicity(df_dedup, group_by='reviewerID')
    print(f"Monotonicity (deduplicated): Global - {results_dedup['global_monotonic']}, Grouped - {results_dedup['grouped_monotonic']}")

    # Load metadata
    try:
        meta_df = pd.read_json(metadata_file, orient='records', lines=True)
    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_file}' not found. Please ensure the processed metadata for {category} is available.")
        return

    # Calculate fraction of ASINs in metadata
    fraction_withdup = calculate_asin_fraction(df_withdup, meta_df)
    print(f"Fraction of ASINs (with duplicates) in metadata: {fraction_withdup:.2%}")
    fraction_dedup = calculate_asin_fraction(df_dedup, meta_df)
    print(f"Fraction of ASINs (deduplicated) in metadata: {fraction_dedup:.2%}")

    # Merge with metadata
    df_withdup = merge_with_metadata(df_withdup, meta_df)
    df_dedup = merge_with_metadata(df_dedup, meta_df)

    # Calculate metrics
    metrics_withdup = calculate_dfmetrics(df_withdup)
    print("\nMetrics (with duplicates):")
    for key, value in metrics_withdup.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    metrics_dedup = calculate_dfmetrics(df_dedup)
    print("\nMetrics (deduplicated):")
    for key, value in metrics_dedup.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    # save df_withdup_merged and df_dedup_merged
    df_withdup.to_json(str(processed_data_dir(f"{category.lower()}2014") / 'df_withdup.json'), orient='records', lines=True)
    df_dedup.to_json(str(processed_data_dir(f"{category.lower()}2014") / 'df_dedup.json'), orient='records', lines=True)

    # Filter reviewers with minimum reviews
    df_withdup_filtered = filter_reviewers_with_min_reviews(df_withdup, min_reviews=3)
    df_dedup_filtered = filter_reviewers_with_min_reviews(df_dedup, min_reviews=3)

    unique_asins_withdup = df_withdup_filtered['asin'].unique()
    unique_asins_dedup = df_dedup_filtered['asin'].unique()
    print(f"\nNumber of unique ASINs (with duplicates, filtered): {len(unique_asins_withdup)}")
    print(f"Number of unique ASINs (deduplicated, filtered): {len(unique_asins_dedup)}")

    # Create corpus metadata
    create_corpus_metadata(meta_df, unique_asins_withdup, output_path=str(processed_data_dir(f"{category.lower()}2014") / 'meta_corpus.json'))



if __name__ == "__main__":
    main()