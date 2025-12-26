import os
import gc
import torch
import json
import pandas as pd
import bm25s
import Stemmer
from typing import List, Dict
from ranx import Qrels, Run, evaluate, fuse
import numpy as np
from datasets import Dataset
from datasets import load_from_disk
from src.utils.project_dirs import get_gen_dir_dataset, processed_data_dir, get_bm25_indexes_dir, get_peruser_metric_dataset_modelname_encoder
from collections import defaultdict
import argparse
from functools import reduce

# --- Dataset Configurations ---
# This dictionary centralizes all dataset-specific information
dataset_configs = {
    "amazon": {
        "user_id_key": "reviewer_id",
        "item_id_key": "asin",
        "meta_columns": ['asin', 'title'],
        "nlang_cols": ['title'],
        "nlang_prefix_map": {'title': 'Title: '},
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

def retrieve_with_topk(genop: List[Dict], retriever: bm25s.BM25,
                       num_return_sequences: int, items_compact: pd.DataFrame,
                       user_id_key: str, item_id_key: str, top_k: int) -> Dict[str, List[Dict]]:
    """
    Retrieves topK items for each generated query and groups by user.
    
    Returns:
        Dict mapping user_id to list of retrieval results, where each result contains:
        - 'query_idx': index of the generated query (0 to num_return_sequences-1)
        - 'items': list of item IDs (topK)
        - 'scores': list of BM25 scores (topK)
        - 'item_indices': list of item indices in items_compact
    """
    l = len(genop)  # number of users
    if num_return_sequences != len(genop[0]['generated_sequences']):
        raise ValueError(f"num_return_sequences {num_return_sequences} "
                         f"does not match the number of generated sequences {len(genop[0]['generated_sequences'])}")
    
    # Flatten all generated queries
    queries_flat = [seq for row in genop for seq in row['generated_sequences']]
    
    # Tokenize and retrieve
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en", show_progress=False)
    res, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    print(f"Retrieved results shape: {res.shape}, scores shape: {scores.shape}")
    
    # Reshape to group by reviewer
    res_reshaped = res.reshape((l, num_return_sequences, top_k))
    scores_reshaped = scores.reshape((l, num_return_sequences, top_k))
    
    # Organize results by user
    user_results = {}
    for i in range(l):
        reviewer_id = genop[i][user_id_key]
        user_results[reviewer_id] = []
        
        for query_idx in range(num_return_sequences):
            item_indices = res_reshaped[i, query_idx]
            item_scores = scores_reshaped[i, query_idx]
            items = items_compact.iloc[item_indices][item_id_key].tolist()
            
            user_results[reviewer_id].append({
                'query_idx': query_idx,
                'items': items,
                'scores': item_scores.tolist(),
                'item_indices': item_indices.tolist()
            })
    
    return user_results

def automated_fusion(user_results_dict: Dict[str, List[Dict]], method="sum",norm="min-max") -> Dict[str, Dict[str, float]]:
    """
    Use ranx.fuse() to automatically aggregate scores across queries for all users.
    
    Creates one run per query (across all users), then fuses them together.
    
    Args:
        user_results_dict: Dict mapping user_id to list of retrieval results.
                         Each result contains 'items', 'scores', and 'query_idx'
    
    Returns:
        Dictionary mapping user_id to aggregated scores: {user_id: {item_id: aggregated_score}}
    """
    # Determine number of queries from first user
    if not user_results_dict:
        return {}
    
    first_user_results = next(iter(user_results_dict.values()))
    num_queries = len(first_user_results)
    
    # Create a run for each query index (across all users)
    # Format: {user_id: {item_id: score}}
    query_runs = []
    
    for query_idx in range(num_queries):
        query_run_dict = {}  # {user_id: {item_id: score}}
        
        for user_id, user_results in user_results_dict.items():
            # Find the result for this query index
            query_result = None
            for result in user_results:
                if result.get('query_idx') == query_idx:
                    query_result = result
                    break
            
            if query_result is None:
                continue
            
            items = query_result['items']
            scores = query_result['scores']
            
            # Create item_id -> score mapping for this user and query
            user_item_scores = {item: score for item, score in zip(items, scores)}
            query_run_dict[user_id] = user_item_scores
        
        # Create Run object for this query (across all users)
        query_run = Run(query_run_dict)
        query_runs.append(query_run)
    
    # Fuse all query runs using sum method with min-max normalization
    fused_run = fuse(
        runs=query_runs,
        norm="min-max",
        method="sum"
    )
    
    # Convert fused Run to dictionary format
    # fused_run.run is {user_id: {item_id: aggregated_score}}
    return fused_run.run

def get_rundict_with_fusion(genop: List[Dict], retriever: bm25s.BM25,
                            num_return_sequences: int, items_compact: pd.DataFrame,
                            user_id_key: str, item_id_key: str, top_k: int, method="sum",norm="min-max") -> Dict[str, Dict[str, float]]:
    """
    Retrieves topK items for each query and uses automated fusion to aggregate scores.
    
    Args:
        genop: List of dictionaries with generated sequences
        retriever: BM25 retriever
        num_return_sequences: Number of generated sequences per user
        items_compact: DataFrame with item metadata
        user_id_key: Key for user ID in genop
        item_id_key: Key for item ID
        top_k: Number of top items to retrieve per query
    
    Returns:
        Dictionary mapping user_id to aggregated item scores: {user_id: {item_id: aggregated_score}}
    """
    # Retrieve topK items for each query
    user_results_dict = retrieve_with_topk(
        genop, retriever, num_return_sequences, items_compact,
        user_id_key, item_id_key, top_k
    )
    
    # Apply automated fusion to aggregate scores across queries
    fused_run_dict = automated_fusion(user_results_dict)
    
    return fused_run_dict

def verify_reviewer_ids(valgen: List[Dict], user_id_key: str) -> None:
    """Verifies the number and uniqueness of reviewer IDs."""
    reviewers = [row[user_id_key] for row in valgen]
    print(f"Number of reviewers: {len(reviewers)}")
    print(f"Number of unique reviewers: {len(set(reviewers))}")

def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int,
                       items_compact: pd.DataFrame, at_k: int, config: Dict, top_k: int = 10, method="sum", norm="min-max"):
    """Evaluates the retrieval performance using rank fusion."""
    qrels = Qrels(get_qrels(genop, config["user_id_key"], config["item_id_key"]))
    retriever = bm25s.BM25.load(retriever_filepath, load_corpus=False)
    run_dict = get_rundict_with_fusion(genop, retriever, num_return_sequences, items_compact,
                                       config["user_id_key"], config["item_id_key"], top_k)
    rundR = Run(run_dict)
    metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    ans = evaluate(qrels, rundR, metrics)
    return qrels, rundR, ans

def load_data(meta_filepath: str, generated_filepath: str, config: Dict) -> tuple[pd.DataFrame, List[Dict]]:
    """
    Args:
        meta_filepath: Path to the meta_corpus.json file.
        generated_filepath: json file containing reviewer_id, asin, seen_asins, generated_sequences for each reviewer.

    Returns:
        A tuple containing the items compact metadata DataFrame and
        the list of dictionaries containing reviewer_id, asin, seen_asins, generated_sequences for each reviewer.
    """
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
        items_compact['nlang'] = final_nlang_series
    else:
        items_compact['nlang'] = pd.Series("", index=items_compact.index)

    item_dict = items_compact.set_index(item_id_key)['nlang'].to_dict()
    with open(generated_filepath, "r") as f:
        genop = json.load(f)
    print(f"Loaded generated data: type={type(genop)}, first element length={len(genop[0]) if genop else 0}")
    return items_compact, genop

def build_bm25_retriever(corpus_list: List[str], index_path: str):
    """Builds and saves a BM25 retriever, or loads it if it exists."""
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus_list, stopwords="en")

    try:
        retriever = bm25s.BM25.load(index_path, load_corpus=False)
        print(f"Loaded BM25 retriever from {index_path}")
    except FileNotFoundError:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(index_path, corpus=corpus_list)
        print(f"Built and saved BM25 retriever to {index_path}")
    return retriever

def get_metrics(meta_filepath: str, generated_filepath: str,
                retriever_filepath: str, num_sequences: int, at_k: int, dataset_name: str,
                config: Dict, peruser_savepath: str, top_k: int = 10):
    """Main function to load data, evaluate retrieval, and print results."""
    print("Starting the evaluation process...")

    # Load data
    print("Loading data...")
    items_compact, genop = load_data(meta_filepath, generated_filepath, config)

    # Verify reviewer IDs
    print("Verifying reviewer IDs...")
    verify_reviewer_ids(genop, config["user_id_key"])

    # Build or load BM25 retriever if it doesn't exist
    corpus_list = items_compact['nlang'].tolist()
    retriever = build_bm25_retriever(corpus_list, retriever_filepath)

    # Evaluate retrieval with rank fusion
    print("Evaluating retrieval performance with rank fusion...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences,
                                           items_compact, at_k, config, top_k, method='max', norm='min-max') # TODO

    # Save peruser metrics
    perusermetrics = rundR.scores

    df_metrics_list = []
    for metric_name, scores_dict in perusermetrics.items():
        df_metric = pd.DataFrame.from_dict(scores_dict, orient='index', columns=[metric_name])
        df_metrics_list.append(df_metric)

    if df_metrics_list:
        df_metrics = pd.concat(df_metrics_list, axis=1, join='outer')

        user_id_column_name = config.get("user_id_key", "user_id")

        df_metrics.index.name = user_id_column_name
        df_metrics = df_metrics.reset_index()
    else:
        user_id_column_name = config.get("user_id_key", "user_id")
        df_metrics = pd.DataFrame(columns=[user_id_column_name])

    if not df_metrics.empty:
        df_metrics.to_json(peruser_savepath + ".jsonl", orient="records", lines=True)
        print(f"Per-user metrics saved to {peruser_savepath}.jsonl")
    else:
        print("No per-user metrics to save.")

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {dataset_name}")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print(f"Top-K retrieval depth: {top_k}")
    print("Metrics:", ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences with rank fusion.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Specific dataset name (e.g., 'beauty', 'sports').")
    parser.add_argument("--data_family", type=str, required=True, choices=["amazon"], help="Family of the dataset.")
    parser.add_argument("--generated_file", type=str, required=True, help="The JSON file containing generated sequences (e.g., 'val_gen_op.json').")
    parser.add_argument("--retriever_index", type=str, required=True, help="The BM25 retriever index file (e.g., 'amznbeauty2014_index').")
    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")
    parser.add_argument("--split", type=str, required=True, help="The split to evaluate on (e.g., 'validation', 'test').")
    parser.add_argument("--short_model_name", type=str, required=True, help="The short model name (e.g., 'llama-1b').")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top items to retrieve per query for rank fusion (default: 10).")

    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    data_family = args.data_family.lower()

    # --- Select the configuration based on the data_family ---
    current_config = None
    if data_family == "amazon":
        current_config = dataset_configs["amazon"].copy()
    else:
        raise ValueError(f"Unsupported data_family: '{data_family}'. Please define its configuration.")

    # --- Construct file paths using the selected configuration and dataset_name ---
    generated_filepath = str(get_gen_dir_dataset(dataset_name) / args.generated_file)
    meta_filepath = str(processed_data_dir(dataset_name) / 'meta_corpus.json')
    retriever_filepath = str(get_bm25_indexes_dir() / args.retriever_index)
    at_k = args.num_sequences

    filename = (args.dataset_name + "_" + args.split + "_" + args.short_model_name)
    encoder_name = "bm25s"
    peruser_savepath = str(get_peruser_metric_dataset_modelname_encoder(args.dataset_name, args.short_model_name, encoder_name) / filename)

    get_metrics(meta_filepath, generated_filepath, retriever_filepath,
                args.num_sequences, at_k, dataset_name, current_config, peruser_savepath, args.top_k)

