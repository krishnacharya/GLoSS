from src.utils.project_dirs import get_movielens_raw_dir, processed_data_dir
import urllib.parse
import os
import pandas as pd
import zipfile
##NOTES ON ML100K
    # u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
    #             Each user has rated at least 20 movies.  Users and items are
    #             numbered consecutively from 1.  The data is randomly
    #             ordered. This is a tab separated list of 
    #             user id | item id | rating | timestamp. 
    #             The time stamps are unix seconds since 1/1/1970 UTC 
    #   
    # u.item     -- Information about the items (movies); this is a tab separated
    #             list of
    #             movie id | movie title | release date | video release date |
    #             IMDb URL | unknown | Action | Adventure | Animation |
    #             Children's | Comedy | Crime | Documentary | Drama | Fantasy |
    #             Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
    #             Thriller | War | Western |
    #             The last 19 fields are the genres, a 1 indicates the movie
    #             is of that genre, a 0 indicates it is not; movies can be in
    #             several genres at once.
    #             The movie ids are the ones used in the u.data data set.

# def ml_100k_processing(ml100k_url:str):
#     data_dir = get_movielens_raw_dir()
#     parsed_reviews_url = urllib.parse.urlparse(ml100k_url)
#     reviews_filename_compressed = os.path.basename(parsed_reviews_url.path)
#     reviews_filename_uncompressed, ext = os.path.splitext(reviews_filename_compressed)
#     if ext == '.gz' or ext == '.zip':
#         reviews_filename_uncompressed = reviews_filename_uncompressed
#     else:
#         reviews_filename_uncompressed = reviews_filename_compressed

#     reviews_file_compressed = str(data_dir / reviews_filename_compressed)
#     reviews_file_uncompressed = str(data_dir / reviews_filename_uncompressed)

#     # Download and unzip the reviews file if it doesn't exist
#     if not os.path.exists(reviews_file_uncompressed):
#         print(f"Downloading reviews from: {ml100k_url} to {reviews_file_compressed}...")
#         os.system(f"wget '{ml100k_url}' -O '{reviews_file_compressed}'")
#         if os.path.exists(reviews_file_compressed) and (reviews_filename_compressed.endswith('.gz') or reviews_filename_compressed.endswith('.zip')):
#             print(f"Unzipping {reviews_file_compressed} to {reviews_file_uncompressed}...")
#             os.system(f"gunzip '{reviews_file_compressed}' -c > '{reviews_file_uncompressed}'")
#         elif not reviews_filename_compressed.endswith('.gz'):
#             reviews_file_uncompressed = reviews_file_compressed
#         elif not os.path.exists(reviews_file_compressed):
#             print(f"Error: Could not download reviews file from {ml100k_url}.")
#             return
#     elif reviews_filename_compressed.endswith('.gz') and not os.path.exists(reviews_file_compressed):
#         print(f"Warning: Uncompressed reviews file exists at {reviews_file_uncompressed}, but compressed version not found. Assuming uncompressed version is the latest.")


#     metadata_file = str(data_dir / "u.item")
#     metadata_df = pd.read_csv(metadata_file, sep='|', header=None, names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
#     metadata_df['genre'] = metadata_df.apply(lambda row: [col for col in row.index[5:] if row[col] == 1], axis=1) # TODO fix can be multple genre, seperate them by comma!
#     metadata_df = metadata_df[['movie_id', 'title', 'genre']]

#     # Load the ratings data
#     df_ui = pd.read_csv(str(data_dir / "u.data"), sep='\t', header=None, names=["user_id", "movie_id", "rating", "timestamp"])
#     # get the number of times each a user has rated, and the number of times each movie has been rated
#     print(df_ui['user_id'].value_counts().describe())
#     print(df_ui['movie_id'].value_counts().describe())

# def calculate_dfmetrics(df):
#     """Calculates dataframe metrics."""
#     unique_users = df['user_id'].nunique()
#     unique_items = len(set(df['movie_id']))
#     total_interactions = df.shape[0]
#     avginteractions_per_user = df['user_id'].value_counts().mean()
#     if unique_items > 0:
#         avginteractions_for_item = total_interactions / unique_items
#         density = total_interactions / (unique_users * unique_items)
#     else:
#         avginteractions_for_item = 0
#         density = 0

#     return {
#         "unique_users": unique_users,
#         "unique_items": unique_items,
#         "total_interactions": total_interactions,
#         "avg_interactions_per_user": avginteractions_per_user,
#         "avg_interactions_per_item": avginteractions_for_item,
#         "density": density
#     }

def merge_with_metadata(df, meta, merge_on='movie_id'):
    merged_df = df.merge(meta, on=merge_on, how='inner')
    return merged_df

def fraction_movieid_in_metadata(df_ui, metadata_df):
    '''
        Returns the fraction of movie_ids in df_ui that are present in metadata_df.
    '''
    unique_movieids_df = set(df_ui['movie_id'])
    unique_movieids_meta = set(metadata_df['movie_id'])
    if not unique_movieids_df:
        return 0.0
    fraction_in_meta = len(unique_movieids_df.intersection(unique_movieids_meta)) / len(unique_movieids_df)
    return fraction_in_meta

def create_corpus_metadata(meta, movieid_list, output_path):
    """Creates a metadata file containing only items present in the provided movie_id list."""
    corpus_items = meta[meta['movie_id'].isin(movieid_list)]
    corpus_items.to_json(output_path, orient='records', lines=True)
    print(f"Corpus metadata saved to {output_path}")

def dedup(df_ui, cols_to_check):
    '''
        Deduplicates any repeated rows in dataframe based on the columns specified in cols_to_check.
        cols_to_check is usually user_id, movie_id, rating, timestamp. if a user has exactly all of these same its an exact duplicate.
        returns the deduplicated dataframe.

        Typically the movielens datasets are very clean, so this is not needed.
    '''
    df_withdup = df_ui.copy()
    df_dedup = df_withdup.drop_duplicates(subset=cols_to_check)
    percentage_duplicates = (1 - len(df_dedup) / len(df_withdup)) * 100
    print(f"Percentage of duplicates: {percentage_duplicates:.2f}%")
    print(f"Shape before deduplication: {df_withdup.shape}")
    print(f"Shape after deduplication: {df_dedup.shape}")
    return df_dedup

def check_monotonicity(df, column='timestamp', group_by=None):
    """Checks if a column is monotonically increasing globally and within groups."""
    results = {}
    results['global_monotonic'] = df[column].is_monotonic_increasing
    if group_by:
        results['grouped_monotonic'] = df.groupby(group_by, observed=False)[column].apply(lambda x: x.is_monotonic_increasing).all()
    else:
        results['grouped_monotonic'] = None
    return results

def ml_100k_processing(ml100k_url: str):
    """
    Downloads, unzips, and processes the MovieLens 100k dataset.

    Args:
        ml100k_url (str): URL of the MovieLens 100k dataset.
    """
    data_dir = get_movielens_raw_dir()
    parsed_reviews_url = urllib.parse.urlparse(ml100k_url)
    reviews_filename_compressed = os.path.basename(parsed_reviews_url.path)

    reviews_file_compressed = str(data_dir / reviews_filename_compressed)

    # Download and unzip the reviews file if it doesn't exist
    if not os.path.exists(reviews_file_compressed):
        print(f"Downloading reviews from: {ml100k_url} to {reviews_file_compressed}...")
        os.system(f"wget '{ml100k_url}' -O '{reviews_file_compressed}'")
        if not os.path.exists(reviews_file_compressed):
            print(f"Error: Could not download reviews file from {ml100k_url}.")
            return

    # Extract the contents of the zip file.  Crucially, we handle the multiple files.
    if reviews_filename_compressed.endswith('.zip'):
        print(f"Unzipping {reviews_file_compressed} to {data_dir}...")
        try:
            with zipfile.ZipFile(reviews_file_compressed, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        except zipfile.BadZipFile:
            print(f"Error: {reviews_file_compressed} is not a valid zip file.")
            return
    else:
        print(f"Error: Expected a .zip file, but got {reviews_filename_compressed}")
        return

    # Define the directory where the files are located (after extraction)
    extracted_data_dir = os.path.join(data_dir, "ml-100k")

    #############################################################
    # Load the metadata
    metadata_file = str(os.path.join(extracted_data_dir, "u.item"))
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file 'u.item' not found in the extracted files at {metadata_file}.")
        return
    try:
        metadata_df = pd.read_csv(metadata_file, sep='|', header=None,
                                    names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown",
                                           "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                                           "Thriller", "War", "Western"],
                                    encoding='latin-1')  # Important: Specify encoding
    except FileNotFoundError:
        print(f"Error: Could not read metadata file {metadata_file}.")
        return

    # Create a list of genres for each movie, comma separated.
    def get_genres(row):
        genres = [col for col in row.index[5:] if row[col] == 1]
        return ','.join(genres) if genres else ''

    metadata_df['genre'] = metadata_df.apply(get_genres, axis=1)
    metadata_df = metadata_df[['movie_id', 'title', 'genre']]

    #############################################################
    # Load the ratings data
    ratings_file = str(os.path.join(extracted_data_dir, "u.data"))
    if not os.path.exists(ratings_file):
        print(f"Error: Ratings file 'u.data' not found in the extracted files at {ratings_file}.")
        return
    try:
        df_ui = pd.read_csv(ratings_file, sep='\t', header=None,
                            names=["user_id", "movie_id", "rating", "timestamp"])
    except FileNotFoundError:
        print(f"Error: Could not read ratings file {ratings_file}.")
        return
    # get the number of times each a user has rated, and the number of times each movie has been rated
    print("User rating counts:")
    print(df_ui['user_id'].value_counts().describe())
    print("Movie rating counts:")
    print(df_ui['movie_id'].value_counts().describe())

    # sort the ratings by user_id and timestamp
    df_ui = df_ui.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    df_ui_dedup = dedup(df_ui, cols_to_check=['user_id', 'movie_id', 'rating', 'timestamp'])
    

    # check monotonicity
    mon_results = check_monotonicity(df_ui, group_by='user_id')
    print(f"Monotonicity  {mon_results['global_monotonic']}, Grouped - {mon_results['grouped_monotonic']}")

    # check fraction of movie_ids in metadata
    frac_movieid_in_metadata = fraction_movieid_in_metadata(df_ui, metadata_df)
    print(f"Fraction of movie_ids in metadata: {frac_movieid_in_metadata}")

    df_ui = merge_with_metadata(df_ui, metadata_df)
    df_ui.to_json(str(processed_data_dir(f"ml100k") / 'df_ui.json'), orient='records', lines=True)

    output_path=str(processed_data_dir(f"ml100k") / 'meta_corpus.json')
    create_corpus_metadata(metadata_df, movieid_list=df_ui['movie_id'].unique(), output_path=output_path)
    

def ml_1m_processing():
    pass


if __name__ == "__main__":
    ml100k_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    ml_100k_processing(ml100k_url)