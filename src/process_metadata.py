import ast
import json
import pandas as pd
from io import StringIO
from tqdm import tqdm
import warnings
import os
import argparse
from src.utils.project_dirs import get_reviews_raw2014_dir, processed_data_dir
import urllib.parse
from pathlib import Path

# https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

def max_word_length_column(df, column_name):
    """
    Find the maximum word-wise length of a column in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to analyze.

    Returns:
        int: The maximum number of words in any entry of the column.
    """
    return df[column_name].apply(lambda x: len(x.split()) if isinstance(x, str) else 0).max()

def load_and_fix_json_lines(file_path):
    """
    Reads a JSON lines file, attempts to fix potential formatting issues,
    and returns a pandas DataFrame. Stops at the first unfixable line.

    Args:
        file_path (str): The path to the JSON lines file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame if successful, None otherwise.
    """
    fixed_lines = []
    try:
        with open(file_path, 'r') as infile:
            total_lines = sum(1 for _ in infile)

        with open(file_path, 'r') as infile:
            for line in tqdm(infile, total=total_lines, desc="Processing lines"):
                cleaned_line = line.strip()
                if cleaned_line:
                    try:
                        python_dict = ast.literal_eval(cleaned_line)
                        json_line = json.dumps(python_dict)
                        fixed_lines.append(json_line)
                    except (SyntaxError, ValueError) as e:
                        print(f"\nError parsing line as Python dictionary: '{cleaned_line}' - Error: {e}")
                        return None  # Stop at the first problematic line

        if fixed_lines:
            fixed_json_string = '\n'.join(fixed_lines)
            return pd.read_json(StringIO(fixed_json_string), orient='records', lines=True)
        else:
            print("No valid lines found to process.")
            return None

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except ValueError as e:
        print(f"ValueError during pandas read: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def handle_duplicate_asins(df):
    """
    Identifies and handles duplicate ASINs in the DataFrame.
    If duplicates have different values in other columns, it prints a warning
    and the differing rows. Otherwise, it removes duplicate ASINs, keeping the first occurrence.

    Args:
        df (pd.DataFrame): The input DataFrame with an 'asin' column.

    Returns:
        pd.DataFrame: The DataFrame with handled duplicate ASINs.
    """
    if 'asin' not in df.columns:
        print("Warning: 'asin' column not found.")
        return df

    print(f"Shape before duplicate handling: {df.shape}, Unique ASINs: {df['asin'].nunique()}")

    def are_rows_equal(group):
        subset = group.iloc[:, 1:] # columns except 'asin'
        return subset.drop_duplicates(keep='first').shape[0] > 1

    # Find ASINs that repeat more than once
    repeated_asins = df['asin'].value_counts()[df['asin'].value_counts() > 1].index
    is_repeated = df['asin'].isin(repeated_asins) # boolean mask indicating rows with repeated ASINs
    different_duplicates = df[is_repeated].groupby('asin', observed=False).apply(are_rows_equal) # first column is the group by i.e. asin

    # Filter for ASINs where the rows are different
    different_asins = different_duplicates[different_duplicates].index.tolist()

    if different_asins:
        print("Warning, ASINs with different values in other columns:")
        for asin in different_asins:
            print(df[df['asin'] == asin])
    else:
        print("All duplicate ASINs have identical values in other columns.")
        df = df.drop_duplicates(subset=['asin'], keep='first').reset_index(drop=True) # Added reset_index

    print(f"Shape after duplicate handling: {df.shape}, Unique ASINs: {df['asin'].nunique()}")
    return df

def inspect_and_clean_columns(df, columns_to_keep):
    """
    Inspects data types and non-empty entries in specified columns,
    handles missing titles (dropping rows), and replaces empty titles with "Unknown".

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_keep (list): A list of column names to keep.

    Returns:
        pd.DataFrame: The cleaned DataFrame with selected columns.
    """
    df = df[columns_to_keep].copy()  # Select and create a copy to avoid SettingWithCopyWarning

    def count_unique_dtypes_column(series):
        return set(type(value) for value in series)

    unique_dtypes = {col: count_unique_dtypes_column(df[col]) for col in df.columns}
    print("Unique data types per column:")
    print(unique_dtypes)

    float_titles_df = df[df['title'].apply(lambda x: isinstance(x, float))]
    if not float_titles_df.empty:
        print("Rows where 'title' is a float:")
        print(float_titles_df)
        df.dropna(subset=['title'], inplace=True)

    def count_non_empty(series):
        def is_non_empty(value):
            if isinstance(value, str):
                return len(value) > 0
            elif isinstance(value, list):
                return len(value) > 0
            else:
                return pd.notnull(value)
        return series.apply(is_non_empty).sum()

    non_empty_counts = {col: count_non_empty(df[col]) for col in df.columns}
    print("\nNon-empty entries per column:")
    print(non_empty_counts)

    if 'title' in df.columns:
        df['title'] = df['title'].replace('', 'Unknown')

    return df

def truncate_text_columns(df, truncation_config):
    """
    Truncates text entries in specified columns to a maximum number of words.

    Args:
        df (pd.DataFrame): The input DataFrame.
        truncation_config (dict): A dictionary where keys are column names and
                                   values are the maximum number of words.

    Returns:
        pd.DataFrame: The DataFrame with truncated text columns.
    """
    def truncate_entry(entry, max_words, column_name):
        if isinstance(entry, str):
            return ' '.join(entry.split()[:max_words])
        else:
            warnings.warn(f"Non-string value encountered in column '{column_name}': {entry}")
            return entry

    for column_name, max_words in truncation_config.items():
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(lambda x: truncate_entry(x, max_words, column_name))
        else:
            warnings.warn(f"Column '{column_name}' not found in the DataFrame.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Process Amazon metadata files.")
    parser.add_argument("--url", type=str, required=True, help="The URL of the metadata JSON.gz file to download.")
    parser.add_argument("--category", type=str, required=True, help="The category of the metadata (used for output directory naming).")
    args = parser.parse_args()
    download_url = args.url
    category = args.category.lower().capitalize()

    raw_dir2014 = get_reviews_raw2014_dir() # this is a Pathlib object

    # Extract filename from URL
    parsed_url = urllib.parse.urlparse(download_url)
    filename_compressed = os.path.basename(parsed_url.path)
    filename_uncompressed_base, ext = os.path.splitext(filename_compressed)
    if ext == '.gz':
        filename_uncompressed = filename_uncompressed_base
    else:
        filename_uncompressed = filename_compressed

    file_path_compressed = str(raw_dir2014 / filename_compressed)
    file_path_uncompressed = str(raw_dir2014 / filename_uncompressed)
    output_dir = processed_data_dir(f"{category.lower()}")
    output_path = str(output_dir / 'meta_proc.json')
    columns_to_keep = ['asin', 'title']
    title_word_limit = 25

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download and unzip the file if it doesn't exist
    if not os.path.exists(file_path_uncompressed):
        print(f"Downloading from: {download_url} to {file_path_compressed}...")
        os.system(f"wget '{download_url}' -O '{file_path_compressed}'") # Enclose URL in quotes

        if os.path.exists(file_path_compressed) and filename_compressed.endswith('.gz'):
            print(f"Unzipping {file_path_compressed} to {file_path_uncompressed}...")
            os.system(f"gunzip '{file_path_compressed}' -c > '{file_path_uncompressed}'") # Use -c to output to new file
        elif not filename_compressed.endswith('.gz'):
            # If it wasn't gzipped, just use the downloaded file
            file_path_uncompressed = file_path_compressed
        elif not os.path.exists(file_path_compressed):
            print(f"Error: Could not download file from {download_url}.")
            return
    elif filename_compressed.endswith('.gz') and not os.path.exists(file_path_compressed):
        print(f"Warning: Uncompressed file exists at {file_path_uncompressed}, but compressed version not found. Assuming uncompressed version is the latest.")

    meta_df = load_and_fix_json_lines(file_path_uncompressed)
    columns_to_keep = ['asin', 'title'] # hack works only for amazon datasets
    meta_df = meta_df[columns_to_keep]    

    if meta_df is not None:
        print(f"Initial shape: {meta_df.shape}, Unique ASINs: {meta_df['asin'].nunique()}")

        meta_df = handle_duplicate_asins(meta_df)
        print(f"Shape after handling duplicates: {meta_df.shape}, Unique ASINs: {meta_df['asin'].nunique()}")

        meta_df = inspect_and_clean_columns(meta_df, columns_to_keep)

        truncation_config = {'title': title_word_limit}
        meta_df = truncate_text_columns(meta_df, truncation_config)

        print(f"Final shape: {meta_df.shape}")
        meta_df.to_json(output_path, orient='records', lines=True)
        print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main()