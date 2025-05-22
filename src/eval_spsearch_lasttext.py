import json
import random
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import bm25s
import Stemmer
from ranx import Qrels, Run, evaluate
from src.utils.project_dirs import processed_data_dir, get_hfdata_dir, get_bm25_indexes_dir, get_peruser_metric_encoder_LIS
import argparse
import os
# SPARSE SEARCH on last text

# --- Dataset Configurations ---
# This dictionary centralizes dataset-specific information
dataset_configs = {
    "amazon": {
        "user_id_key": "reviewer_id",
        "item_id_key": "asin",
        "meta_columns": ['asin', 'title'],
        "nlang_cols": ['title'],
        "nlang_prefix_map": {'title': 'Title: '},
    },
    "movielens": {
        "user_id_key": "user_id",
        "item_id_key": "movie_id",
        "meta_columns": ['movie_id', 'title', 'genre'],
        "nlang_cols": ['title', 'genre'],
        "nlang_prefix_map": {'title': 'Title: ', 'genre': 'Genres: '},
    },
}


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

def evaluate_model(qrels: Qrels, run: Run, metrics: List[str], model_name: str):
    """Evaluates a recommendation model and prints the results."""
    print(f"\nEvaluating {model_name}:")
    print("-" * 30)
    results = evaluate(qrels, run, metrics)
    print(results)
    print("-" * 30)
    return results




def prepare_metadata_for_retrieval(meta_corpus_path: str, config: Dict) -> (pd.DataFrame, Dict[str, str]):
    """
    Prepares metadata for retrieval, handling both Amazon and MovieLens datasets.

    Args:
        meta_corpus_path: Path to the meta corpus file.
        config: Dataset-specific configuration dictionary.

    Returns:
        A tuple containing the compact metadata DataFrame and a dictionary mapping item IDs to natural language descriptions.
    """
    meta_corpus = pd.read_json(meta_corpus_path, orient='records', lines=True)
    meta_corpus = meta_corpus.astype(str)

    # Use dataset-specific column names
    item_id_key = config["item_id_key"]
    nlang_cols = config["nlang_cols"]
    nlang_prefix_map = config["nlang_prefix_map"]

    # Apply column renaming from config if specified
    if 'meta_columns' in config and np.all((config['meta_columns']) == (meta_corpus.columns)):
        meta_corpus.columns = config['meta_columns']
    elif 'meta_columns' in config and len(config['meta_columns']) != len(meta_corpus.columns):
        print(f"Warning: config['meta_columns'] length ({len(config['meta_columns'])}) does not match actual meta_corpus columns length ({len(meta_corpus.columns)}). "
              f"Proceeding without automatic column renaming based on config. Ensure your meta_corpus has the correct column names: {config['meta_columns']}")

    items_compact = meta_corpus[[item_id_key]].copy()

    # Construct the 'nlang' column dynamically
    nlang_parts_series = []
    for col in nlang_cols:
        prefix = nlang_prefix_map.get(col, "")
        nlang_parts_series.append(prefix + meta_corpus[col].astype(str))

    if nlang_parts_series:
        final_nlang_series = nlang_parts_series[0]
        for i in range(1, len(nlang_parts_series)):
            final_nlang_series = final_nlang_series.str.cat(nlang_parts_series[i], sep=", ")
        items_compact['nlang'] = final_nlang_series
    else:
        items_compact['nlang'] = pd.Series("", index=items_compact.index)


    item_id_to_nlang = items_compact.set_index(item_id_key)['nlang'].to_dict()
    return items_compact, item_id_to_nlang



def get_qrels(dataset: Dataset, config: Dict) -> Dict[str, Dict[str, int]]:
    """
    Generates ground truth (qrels) from the dataset, compatible with both Amazon and MovieLens.
    Args:
        dataset: The dataset.
        config: the dataset config.
    Returns:
        A dictionary representing the qrels.
    """
    qrels_dict = {}
    user_id_key = config["user_id_key"]
    item_id_key = config["item_id_key"]
    for row in dataset:
        user_id = row[user_id_key]
        item_id = row[item_id_key]
        if user_id not in qrels_dict:
            qrels_dict[user_id] = {}
        qrels_dict[user_id][item_id] = 1
    return qrels_dict



def textbased_lastsimilar(dataset: Dataset, retriever: bm25s.BM25, \
                          item_id_to_nlang: Dict[str, str], items_compact: pd.DataFrame, \
                          config: Dict, k: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Generates recommendations based on text similarity of the last seen item using BM25,
    compatible with both Amazon and MovieLens.
    Args:
        dataset: The dataset.
        retriever: The BM25 retriever.
        item_id_to_nlang: Dictionary mapping item IDs to natural language descriptions.
        items_compact: DataFrame containing item IDs and other metadata.
        config: Dataset-specific configuration.
        k: The number of recommendations to generate.
    Returns:
        A dictionary of recommendations.
    """
    l = len(dataset)
    user_id_key = config["user_id_key"]
    item_id_key = config["item_id_key"]
    queries_flat = [item_id_to_nlang[seen[-1]] for seen in dataset['seen_' + item_id_key + 's']]
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en")
    res, scores = retriever.retrieve(query_tokens, k=k)
    run_dict = {}
    for i in range(l):
        user_id = dataset[user_id_key][i]
        item_indices = res[i]
        item_scores = scores[i]
        items = items_compact.iloc[item_indices][item_id_key].tolist()
        run_dict[user_id] = {item: score for item, score in zip(items, item_scores)}
    return run_dict

def main(dataset_name: str, dataset_path: str, meta_corpus_path: str, bm25_index_name: str, \
         k: int = 5, metrics: List[str] = ["recall@5", "ndcg@5", "mrr"], \
         dataset_split: str = "test", data_family: str = "movielens"):
    """
    Main function to run the text-based last similar item recommendation evaluation.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        dataset_path: Path to the Hugging Face dataset.
        meta_corpus_path: Path to the meta corpus file.
        bm25_index_name: Name of the BM25 index.
        k: The number of recommendations to generate.
        metrics: List of evaluation metrics to use.
        dataset_split: The dataset split to evaluate (e.g., 'train', 'validation', 'test').
        data_family: The family of the dataset, either "amazon" or "movielens".
    """
    if data_family not in dataset_configs:
        raise ValueError(f"Unsupported data_family: {data_family}.  Must be one of {list(dataset_configs.keys())}")
    config = dataset_configs[data_family]

    dataset = load_from_disk(dataset_path)
    items_compact, item_id_to_nlang = prepare_metadata_for_retrieval(meta_corpus_path, config)
    
    corpus_list = items_compact['nlang'].tolist()

    # Build or load BM25 retriever
    bm25_index_path = str(get_bm25_indexes_dir() / bm25_index_name)
    retriever = build_bm25_retriever(corpus_list, bm25_index_path)
    
    # Prepare ground truth
    qrels_test = Qrels(get_qrels(dataset[dataset_split], config))
    print(f"\nQrels ({dataset_split}):")
    print("-" * 30)

    # Evaluate Text-Based Last Similar Recommendations
    run_test_textbased_lastsimilar = Run(textbased_lastsimilar(dataset[dataset_split], retriever, item_id_to_nlang, items_compact, config, k=k)) # Pass config
    ans = evaluate_model(qrels_test, run_test_textbased_lastsimilar, metrics, "Text-Based Last Similar Recommendations")
    perusermetrics = run_test_textbased_lastsimilar.scores
    peruser_savepath = str(get_peruser_metric_encoder_LIS("bm25") / f"{dataset_name}.jsonl")

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
        df_metrics.to_json(peruser_savepath, orient="records", lines=True)
        print(f"Per-user metrics saved to {peruser_savepath}")
    else:
        print("No per-user metrics to save.")

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {args.dataset}")
    print(f"BM25 index file: {bm25_index_path}")
    print("Metrics:", ans)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation models on a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="The path to the Hugging Face dataset.")
    parser.add_argument("--bm25_index_name", type=str, required=True, help="The name of the BM25 index.")
    parser.add_argument("--k", type=int, default=5, help="The value of k for evaluation metrics (e.g., recall@k).")
    parser.add_argument("--metrics", nargs='+', default=["recall@5", "ndcg@5", "mrr"], help="List of evaluation metrics to use.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate (e.g., 'train', 'validation', 'test').")
    parser.add_argument("--data_family", type=str, required=True, choices=["amazon", "movielens"], help="The family of the dataset: 'amazon' or 'movielens'.")

    args = parser.parse_args()

    hf_dir = get_hfdata_dir()
    processed_dir = processed_data_dir(args.dataset)
    bm25_index_name = args.bm25_index_name
    dataset_path = str(hf_dir / args.dataset)
    meta_corpus_path = str(processed_dir / 'meta_corpus.json')
    data_family = args.data_family

    main(args.dataset, dataset_path, meta_corpus_path, bm25_index_name, args.k, args.metrics, args.split, data_family)

# LIS on beauty with bm25s
# Metrics: {'recall@5': 0.04333050127442651, 'ndcg@5': 0.023179244762936268, 'mrr': 0.016472894215147044}

# LIS on toys with bm25s
# Metrics: {'recall@5': 0.05212683681361176, 'ndcg@5': 0.026660292010049123, 'mrr': 0.0182822033170061}

# LIS on sports with bm25s
# Metrics: {'recall@5': 0.021153467988875466, 'ndcg@5': 0.011012792672956575, 'mrr': 0.007661226133288385}