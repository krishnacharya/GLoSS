from src.utils.project_dirs import get_movielens_raw_dir, processed_data_dir, get_hfdata_dir
import urllib.parse
import os
import pandas as pd
import zipfile
from collections import defaultdict
import random
from datasets import Dataset, DatasetDict
from pathlib import Path
from src.process_metadata import truncate_text_columns

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

# NOTES ON ML1M
    #ratings.dat has the following format: user_id::movie_id::rating::timestamp

    # All ratings are contained in the file "ratings.dat" and are in the
    # following format:

    # UserID::MovieID::Rating::Timestamp

    # - UserIDs range between 1 and 6040 
    # - MovieIDs range between 1 and 3952
    # - Ratings are made on a 5-star scale (whole-star ratings only)
    # - Timestamp is represented in seconds since the epoch as returned by time(2)
    # - Each user has at least 20 ratings

    # Movie information is in the file "movies.dat" and is in the following
    # format:

    # MovieID::Title::Genres

    # - Titles are identical to titles provided by the IMDB (including
    # year of release)
    # - Genres are pipe-separated and are selected from the following genres:

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


def create_hf_dataset_MLFamily(df_ui, meta_corpus, save_path:str, Nval:int=512, random_seed:int=42, dname = 'ml100k'):
    '''
        Makes a huggingface dataset for movielens family of datasets.

        meta corpus should have columns: movie_id, title, genre
        df_ui should have columns: user_id, movie_id, rating, timestamp, title, genre
    '''
    def group_movies_for_user(df_ui:pd.DataFrame):
        '''
            Group movie ids by user id from the interaction dataframe.
            Args:
                df_ui_path (str): Path to the interactions with 'user_id' and 'movie_id' columns.
            Returns:
                Dict[str, List[str]]: A dictionary mapping user IDs to lists of movie IDs.
        '''
        user_movies = defaultdict(list)
        for _, row in df_ui.iterrows():
            user_movies[row['user_id']].append(row['movie_id'])
        return user_movies
    
    assert dname in ['ml100k', 'ml1m'], "dname must be one of 'ml100k', 'ml1m'"
    assert meta_corpus.columns.tolist() == ['movie_id', 'title', 'genre']
    assert df_ui.columns.tolist() == ['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'genre']
    
    movies_compact = meta_corpus[['movie_id']].copy()
    movies_compact['nlang'] = "Title: " + meta_corpus['title'] + ", Genres: " + meta_corpus['genre']
    movieid_to_nlang = movies_compact.set_index('movie_id')['nlang'].to_dict()

    # Group 
    user_movies = group_movies_for_user(df_ui)

    # Define prompt template
    movielens_prompt = (
        "Below is a user's Movielens watch history in chronological order (earliest to latest). \n"
        "Each movie is represented by the following format: Title: <movie title>, Genres: <movie genres> \n"
        "Based on this history, predict **only one** movie the user is most likely to watch next in the same format.\n\n"
        "### Watch history:\n"
        "{}\n\n"
        "### Next movie:\n"
        "{}"
    )

    # Split reviewers into validation and others
    random.seed(random_seed)
    all_users = list(user_movies.keys())
    Nval = min(Nval, len(all_users) // 2)
    val_users = set(random.sample(all_users, Nval))

    train_records, val_records, test_records = [], [], []
    for user_id, movies_list in user_movies.items():
        n = len(movies_list)
        if n < 3:
            continue # Skip users with less than 3 movies though in movielens family of datasets, users have at least 20 interactions.

        # raise error if movie_id not in movieid_to_nlang
        formatted_movies_list = [movieid_to_nlang.get(movie_id, f"Unknown Movie: {movie_id}") for movie_id in movies_list] 
        
        # Prepare test set for all reviewers
        test_ptext = movielens_prompt.format("\n".join(formatted_movies_list[:n-1]), "")
        test_text = movielens_prompt.format("\n".join(formatted_movies_list[:n-1]), formatted_movies_list[n-1])

        test_seen_movids = movies_list[:n-1]
        test_movid = movies_list[n-1]
        test_mov_text = formatted_movies_list[n-1]
        
        test_records.append({
            "user_id": user_id,
            "ptext": test_ptext,
            "text": test_text,
            "seen_movie_ids": test_seen_movids,
            "movie_id": test_movid,
            "movie_id_text": test_mov_text
        })

        if user_id in val_users:  # Validation set
            val_ptext = movielens_prompt.format("\n".join(formatted_movies_list[:n-2]), "")
            val_text = movielens_prompt.format("\n".join(formatted_movies_list[:n-2]), formatted_movies_list[n-2])
            val_seen_movids = movies_list[:n-2]
            val_movid = movies_list[n-2]
            val_mov_text = formatted_movies_list[n-2]
            val_records.append({
                "user_id": user_id,
                "ptext": val_ptext,
                "text": val_text,
                "seen_movie_ids": val_seen_movids,
                "movie_id": val_movid,
                "movie_id_text": val_mov_text
            })
            # Train set is shorter for validation users
            train_text = movielens_prompt.format("\n".join(formatted_movies_list[:n-3]), formatted_movies_list[n-3])
            train_records.append({"user_id": user_id, "text": train_text})
        else:  # Train set for non-validation users
            train_text = movielens_prompt.format("\n".join(formatted_movies_list[:n-2]), formatted_movies_list[n-2])
            train_records.append({"user_id": user_id, "text": train_text})

    # Convert lists to Hugging Face datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_records))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_records))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_records))

    # Create a DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Save dataset locally
    output_path = Path(save_path) / dname
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"Hugging Face datasets saved to {output_path}")

    return dataset_dict

        

def ml_100k_processing(ml100k_url: str, valfrac:float=0.05):
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
                                    encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: Could not read metadata file {metadata_file}.")
        return

    # Create a list of genres for each movie, comma separated.
    def get_genres(row):
        genres = [col for col in row.index[5:] if row[col] == 1]
        return ','.join(genres) if genres else ''

    metadata_df['genre'] = metadata_df.apply(get_genres, axis=1)
    metadata_df = metadata_df[['movie_id', 'title', 'genre']]
    # metadata_df['movie_id'] = metadata_df['movie_id'].astype(str)
    metadata_df = metadata_df.astype(str)

    truncation_config = {'title': 25} # 25 words is a lot for a movie title, but can filter corruptions
    metadata_df = truncate_text_columns(metadata_df, truncation_config)

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

    # df_ui['movie_id'] = df_ui['movie_id'].astype(str)
    # df_ui['user_id'] = df_ui['user_id'].astype(str)
    df_ui = df_ui.astype(str)
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

    hf_dir = get_hfdata_dir()
    Nval = int(valfrac * len(df_ui['user_id'].unique()))
    create_hf_dataset_MLFamily(df_ui, metadata_df, save_path=hf_dir, Nval=Nval, random_seed=42, dname='ml100k')

def ml_1m_processing(ml1m_url: str, valfrac:float=0.05):
    """
    Downloads, unzips, and processes the MovieLens 1M dataset.
    Args:
        ml1m_url (str): URL of the MovieLens 1M dataset.
    """
    data_dir = get_movielens_raw_dir()
    parsed_reviews_url = urllib.parse.urlparse(ml1m_url)
    reviews_filename_compressed = os.path.basename(parsed_reviews_url.path)

    reviews_file_compressed = str(data_dir / reviews_filename_compressed)

    # Download and unzip the reviews file if it doesn't exist
    if not os.path.exists(reviews_file_compressed):
        print(f"Downloading reviews from: {ml1m_url} to {reviews_file_compressed}...")
        os.system(f"wget '{ml1m_url}' -O '{reviews_file_compressed}'")
        if not os.path.exists(reviews_file_compressed):
            print(f"Error: Could not download reviews file from {ml1m_url}.")
            return

    # Extract the contents of the zip file.
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
    extracted_data_dir = os.path.join(data_dir, "ml-1m")

    #############################################################
    # Load the metadata
    metadata_file = str(os.path.join(extracted_data_dir, "movies.dat"))
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file 'movies.dat' not found in the extracted files at {metadata_file}.")
        return
    try:
        metadata_df = pd.read_csv(metadata_file, sep='::', header=None,
                                    names=["movie_id", "title", "genre"],
                                    encoding='latin-1', engine='python') # 'python' engine for 'sep' with multiple chars
    except FileNotFoundError:
        print(f"Error: Could not read metadata file {metadata_file}.")
        return

    # Genres are pipe-separated, replace with comma for consistency
    metadata_df['genre'] = metadata_df['genre'].str.replace('|', ',', regex=False)
    metadata_df = metadata_df.astype(str)

    truncation_config = {'title': 25}
    metadata_df = truncate_text_columns(metadata_df, truncation_config)

    #############################################################
    # Load the ratings data
    ratings_file = str(os.path.join(extracted_data_dir, "ratings.dat"))
    if not os.path.exists(ratings_file):
        print(f"Error: Ratings file 'ratings.dat' not found in the extracted files at {ratings_file}.")
        return
    try:
        df_ui = pd.read_csv(ratings_file, sep='::', header=None,
                            names=["user_id", "movie_id", "rating", "timestamp"],
                            engine='python')
    except FileNotFoundError:
        print(f"Error: Could not read ratings file {ratings_file}.")
        return

    df_ui = df_ui.astype(str)
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
    df_ui.to_json(str(processed_data_dir(f"ml1m") / 'df_ui.json'), orient='records', lines=True)

    output_path=str(processed_data_dir(f"ml1m") / 'meta_corpus.json')
    create_corpus_metadata(metadata_df, movieid_list=df_ui['movie_id'].unique(), output_path=output_path)

    hf_dir = get_hfdata_dir()
    Nval = int(valfrac * len(df_ui['user_id'].unique()))
    create_hf_dataset_MLFamily(df_ui, metadata_df, save_path=hf_dir, Nval=Nval, random_seed=42, dname='ml1m')



if __name__ == "__main__":
    ml100k_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    ml_100k_processing(ml100k_url)

    ml1m_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    ml_1m_processing(ml1m_url)