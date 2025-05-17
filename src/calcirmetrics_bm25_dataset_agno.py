import os
import gc
import torch
import json
import pandas as pd
import bm25s
import Stemmer
from typing import List, Dict
from ranx import Qrels, Run, evaluate
import numpy as np
from datasets import Dataset
from datasets import load_from_disk
from src.utils.project_dirs import get_gen_dir_dataset, processed_data_dir, get_bm25_indexes_dir
from collections import defaultdict
import argparse

# --- Dataset Configurations ---
# This dictionary centralizes all dataset-specific information
dataset_configs = {
    "amazon": { # Base configuration for Amazon-like datasets
        "user_id_key": "reviewer_id",
        "item_id_key": "asin",
        "meta_columns": ['asin', 'title'], # Expected columns for meta_corpus
        "nlang_cols": ['title'],
        "nlang_prefix_map": {'title': 'Title: '},
        # No 'processed_data_dir_name' here; it will be dynamically set by the specific dataset name
    },
    "movielens": {
        "user_id_key": "user_id",
        "item_id_key": "movie_id",
        "meta_columns": ['movie_id', 'title', 'genre'], # Expected columns for meta_corpus
        "nlang_cols": ['title', 'genre'],
        "nlang_prefix_map": {'title': 'Title: ', 'genre': 'Genres: '},
        # No 'processed_data_dir_name' here; it will be dynamically set by the specific dataset name
    }
}

def get_qrels(genop: List[Dict], user_id_key: str, item_id_key: str) -> Dict[str, Dict[str, int]]:
    '''
    Returns: dict[reviewer_id] = {item_id1: 1, item_id2: 1, ...}
    '''
    qrels_dict = defaultdict(dict)
    for row in genop:
        reviewer = row[user_id_key]
        item = row[item_id_key]
        qrels_dict[reviewer][item] = 1
    return dict(qrels_dict)

def get_unique_sorted_items(items: List[str], scores: List[float]) -> Dict[str, float]:
    """
        Deduplicates a list of items and their associated scores,retaining only the highest score for each unique item.
        Items are sorted by their scores in descending order, with the first occurrence of an item (which will have its highest score due to sorting) being kept.
    """
    seen = set()
    item_score_pairs = []
    for item, score in sorted(zip(items, scores), key=lambda x: -x[1]):
        if item not in seen:
            item_score_pairs.append((item, score))
            seen.add(item)
    return dict(item_score_pairs)

def get_rundict(genop: List[Dict], retriever: bm25s.BM25,
    num_return_sequences: int, items_compact: pd.DataFrame, user_id_key: str, item_id_key: str) -> Dict[str, Dict[str, float]]:
    # Changed 'asins_compact' parameter to 'items_compact'
    run_dict = {}
    l = len(genop)
    if num_return_sequences != len(genop[0]['generated_sequences']):
        raise ValueError(f"num_return_sequences {num_return_sequences} "
                         f"does not match the number of generated sequences {len(genop[0]['generated_sequences'])}")
    # Flatten all generated queries
    queries_flat = [seq for row in genop for seq in row['generated_sequences']]
    # Tokenize and retrieve
    stemmer = Stemmer.Stemmer("english")
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en")
    res, scores = retriever.retrieve(query_tokens, k=1)
    print(f"Retrieved results shape: {res.shape}, scores shape: {scores.shape}")
    # Reshape to group by reviewer
    res_temp = res.reshape((l, num_return_sequences))
    scores_temp = scores.reshape((l, num_return_sequences))
    res = res_temp
    scores = scores_temp
    for i in range(l):
        reviewer_id = genop[i][user_id_key]
        item_indices = res[i]
        item_scores = scores[i]
        items = items_compact.iloc[item_indices][item_id_key].tolist()
        run_dict[reviewer_id] = get_unique_sorted_items(items, item_scores)
    return run_dict

def verify_reviewer_ids(valgen: List[Dict], user_id_key: str) -> None:
    """Verifies the number and uniqueness of reviewer IDs."""
    reviewers = [row[user_id_key] for row in valgen]
    print(f"Number of reviewers: {len(reviewers)}")
    print(f"Number of unique reviewers: {len(set(reviewers))}")

def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int,
                       items_compact: pd.DataFrame, at_k: int, config: Dict):
    """Evaluates the retrieval performance."""
    qrels = Qrels(get_qrels(genop, config["user_id_key"], config["item_id_key"]))
    retriever = bm25s.BM25.load(retriever_filepath, load_corpus=False)
    run_dict = get_rundict(genop, retriever, num_return_sequences, items_compact,config["user_id_key"], config["item_id_key"])
    rundR = Run(run_dict)
    metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    ans = evaluate(qrels, rundR, metrics)
    return qrels, rundR, ans

def load_data(meta_filepath: str, generated_filepath: str, config: Dict) -> tuple[pd.DataFrame, List[Dict]]:
    """Loads the meta data and the generated sequences using a config dictionary."""
    meta_corpus = pd.read_json(meta_filepath, orient='records', lines=True)
    meta_corpus = meta_corpus.astype(str)

    # Apply column renaming from config if specified
    if 'meta_columns' in config and np.all((config['meta_columns']) == (meta_corpus.columns)):
        meta_corpus.columns = config['meta_columns']
    elif 'meta_columns' in config and len(config['meta_columns']) != len(meta_corpus.columns):
        print(f"Warning: config['meta_columns'] length ({len(config['meta_columns'])}) does not match actual meta_corpus columns length ({len(meta_corpus.columns)}). "
              f"Proceeding without automatic column renaming based on config. Ensure your meta_corpus has the correct column names: {config['meta_columns']}")

    item_id_key = config["item_id_key"]
    nlang_cols = config["nlang_cols"]
    nlang_prefix_map = config["nlang_prefix_map"]

    # Check if all required nlang_cols exist after potential renaming
    for col in nlang_cols:
        if col not in meta_corpus.columns:
            raise ValueError(f"Required nlang column '{col}' not found in meta_corpus for {meta_filepath}. "
                             f"Available columns: {meta_corpus.columns.tolist()}. Check your 'meta_columns' config or the actual meta file's column names.")

    items_compact = meta_corpus[[item_id_key]].copy()

    # Dynamically build the nlang string based on nlang_cols and nlang_prefix_map
    nlang_parts_series = []
    for col in nlang_cols:
        prefix = nlang_prefix_map.get(col, "")
        # Create a Series with the prefix prepended to each element, ensuring string type
        nlang_parts_series.append(prefix + meta_corpus[col].astype(str))

    # Construct the final 'nlang' column by joining the series with a comma and space
    if nlang_parts_series:
        final_nlang_series = nlang_parts_series[0]
        for i in range(1, len(nlang_parts_series)):
            final_nlang_series = final_nlang_series.str.cat(nlang_parts_series[i], sep=", ")
        items_compact['nlang'] = final_nlang_series # Changed 'asins_compact' to 'items_compact'
    else:
        items_compact['nlang'] = pd.Series("", index=items_compact.index)

    item_dict = items_compact.set_index(item_id_key)['nlang'].to_dict()
    with open(generated_filepath, "r") as f:
        genop = json.load(f)
    print(f"Loaded generated data: type={type(genop)}, first element length={len(genop[0]) if genop else 0}")
    return items_compact, genop

def get_metrics(meta_filepath: str, generated_filepath: str,
                retriever_filepath: str, num_sequences: int, at_k: int, dataset_name: str,
                config: Dict):
    """Main function to load data, evaluate retrieval, and print results."""
    print("Starting the evaluation process...")

    # Load data
    print("Loading data...")
    items_compact, genop = load_data(meta_filepath, generated_filepath, config)

    # Verify reviewer IDs
    print("Verifying reviewer IDs...")
    verify_reviewer_ids(genop, config["user_id_key"])

    # Evaluate retrieval
    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences,
                                           items_compact, at_k, config)

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {dataset_name}")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print("Metrics:", ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Specific dataset name (e.g., 'beauty', 'ml100k', 'sports').")
    parser.add_argument("--data_family", type=str, required=True, choices=["amazon", "movielens"], help="Family of the dataset (e.g., 'amazon', 'movielens').")
    parser.add_argument("--generated_file", type=str, required=True, help="The JSON file containing generated sequences (e.g., 'val_gen_op.json').")
    parser.add_argument("--retriever_index", type=str, required=True, help="The BM25 retriever index file (e.g., 'amznbeauty2014_index').")
    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")

    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    data_family = args.data_family.lower()

    # --- Select the configuration based on the data_family ---
    current_config = None
    if data_family == "amazon":
        current_config = dataset_configs["amazon"].copy()
    elif data_family == "movielens":
        current_config = dataset_configs["movielens"].copy()
    else:
        raise ValueError(f"Unsupported data_family: '{data_family}'. Please define its configuration.")

    # --- Construct file paths using the selected configuration and dataset_name ---
    generated_filepath = str(get_gen_dir_dataset(dataset_name) / args.generated_file)
    meta_filepath = str(processed_data_dir(dataset_name) / 'meta_corpus.json')
    retriever_filepath = str(get_bm25_indexes_dir() / args.retriever_index)
    at_k = args.num_sequences

    get_metrics(meta_filepath, generated_filepath, retriever_filepath,
                args.num_sequences, at_k, dataset_name, current_config)


# Movielens 100k, llama-1b-649 step cpt, TEST
# Metrics: {'recall@5': np.float64(0.05938494167550371), 'ndcg@5': np.float64(0.031014263012511995), 'mrr': np.float64(0.02189819724284199)}

# Movielens 100k, llama-3B-590 step cpt, TEST
# Metrics: {'recall@5': np.float64(0.05832449628844114), 'ndcg@5': np.float64(0.032443425480723265), 'mrr': np.float64(0.02407211028632025)}

# Movielens 100k, llama-8B step cpt, TEST
