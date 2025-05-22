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
from src.utils.project_dirs import processed_data_dir, get_hfdata_dir, get_bm25_indexes_dir, get_dense_retrieval_index_dir, get_peruser_metric_encoder_LIS
from src.eval_baseline_dataset_agno import prepare_metadata_for_retrieval, get_qrels
from src.calcirmetrics_denseret_dataset_agno import create_dense_retrieval_index # TODO refactor to utils

from retriv import DenseRetriever
import argparse
import os

# DENSE SEARCH on last text
dataset_configs = {
    "amazon": {
        "user_id_key": "reviewer_id",
        "item_id_key": "asin",
        "meta_columns": ['asin', 'title'],
        "nlang_cols": ['title'],
        "nlang_prefix_map": {'title': 'Title: '},
    }
}

def get_rundict_from_ssearch(dataset: Dataset, dr: DenseRetriever, item_id_to_nlang: Dict[str, str], \
                              at_k: int, user_id_key: str, item_id_key: str, \
                              batch_size: int = None):
    """
    Batched version of dense retrieval and formatting into a run dictionary.

    Args:
        last_texts: List of last texts.
        dr: The DenseRetriever object.
        at_k: scores for top k items for each
        user_id_key: Key to access user IDs in genop.
        item_id_key: Key to access item IDs.
        batch_size: Batch size for the dense retrieval.

    Returns:
        A dictionary formatted as {user_id: {item_id: score, ...}, ...}.
    """
    if batch_size is None:
        batch_size = num_return_sequences
    
    l = len(dataset)    
    all_queries = []
    for i in range(l):
        user_id = dataset[i][user_id_key]
        last_text = item_id_to_nlang[dataset[i]['seen_' + item_id_key + 's'][-1]] # seen_asins
        all_queries.append({'id': f'{user_id}', 'text': last_text})
    
    # batched search
    res = dr.bsearch(queries=all_queries, cutoff=at_k, batch_size=batch_size,show_progress=False)  # dictionary of reviewer_id_j -> {item_id: score} # Changed asin to item_id
    run_dict = defaultdict(dict)
    for userid, v in res.items():  # each k is a reviewer_id and v is a dictionary of item_id and score
        for item_id, score in v.items():
            run_dict[userid][item_id] = score
    return run_dict

def main(dataset_name: str, dataset_path: str, meta_corpus_path: str, \
         encoder_name: str, k: int = 5, metrics: List[str] = ["recall@5", "ndcg@5", "mrr"],\
           dataset_split: str = "test", data_family: str ="amazon", max_length: int = 256, 
           use_ann: bool = False, use_gpu: bool = True, batch_size: int = 128, show_progress: bool = False):

    if data_family not in dataset_configs:
        raise ValueError(f"Unsupported data_family: {data_family}.  Must be one of {list(dataset_configs.keys())}")
    config = dataset_configs[data_family]

    dataset = load_from_disk(dataset_path)
    items_compact, item_id_to_nlang = prepare_metadata_for_retrieval(meta_corpus_path, config)

    corpus_list = items_compact['nlang'].tolist()

    # retriever index
    retriever_filepath = str(get_dense_retrieval_index_dir(encoder_name) / f"{dataset_name}_index")
    if not os.path.exists(retriever_filepath):
        print("Making dense retrieval index...")
        create_dense_retrieval_index(meta_corpus_path, retriever_filepath, config["item_id_key"],
                                      max_length, encoder_name, use_ann, use_gpu, batch_size,
                                      show_progress, config=config, item_dict=item_id_to_nlang)
    else:
        print("Dense retrieval index already exists...")
        dr = DenseRetriever.load(retriever_filepath)
    
    # evaluate
    peruser_savepath = str(get_peruser_metric_encoder_LIS(encoder_name) / f"{dataset_name}.jsonl")
    qrels = Qrels(get_qrels(dataset[dataset_split], config))
    print(f"\nQrels ({dataset_split}):")
    print("-" * 30)
    rundR = Run(get_rundict_from_ssearch(dataset = dataset[dataset_split], 
                                        dr=dr,
                                        item_id_to_nlang=item_id_to_nlang,
                                        at_k=k,
                                        user_id_key=config["user_id_key"],
                                        item_id_key=config["item_id_key"],
                                        batch_size=batch_size))

    ans = evaluate(qrels, rundR, metrics)
    perusermetrics = rundR.scores

    df_metrics_list = []
    for metric_name, scores_dict in perusermetrics.items():
        df_metric = pd.DataFrame.from_dict(scores_dict, orient='index', columns=[metric_name])
        df_metrics_list.append(df_metric)

    if df_metrics_list:
        df_metrics = pd.concat(df_metrics_list, axis=1, join='outer')
        user_id_column_name = config.get("user_id_key", "user_id") # agnostic mapping
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
    print(f"Dataset: {dataset_name}")
    print(f"Retriever index file: {retriever_filepath}")
    print("Metrics:", ans)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_family", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, required=True)
    parser.add_argument("--k", type=int, default=5, help="The value of k for evaluation metrics (e.g., recall@k).")
    parser.add_argument("--metrics", nargs='+', default=["recall@5", "ndcg@5", "mrr"], help="List of evaluation metrics to use.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate (e.g., 'train', 'validation', 'test').")
    args = parser.parse_args()

    hf_dir = get_hfdata_dir()
    processed_dir = processed_data_dir(args.dataset_name)
    encoder_name = args.encoder_name
    dataset_path = str(hf_dir / args.dataset_name)
    meta_corpus_path = str(processed_dir / 'meta_corpus.json')
    data_family = args.data_family

    main(args.dataset_name, dataset_path, meta_corpus_path, encoder_name, args.k, args.metrics, args.split, data_family)

# on beauty LIR with intfloat/e5-small-v2"
# Metrics: {'recall@5': 0.041810132808657155, 'ndcg@5': 0.02236049700748934, 'mrr': 0.01589008630326879}

# on toys LIR with intfloat/e5-small-v2"

# on sports LIR with intfloat/e5-small-v2"


# ---
# LIS on beauty with intfloat/e5-base-v2"

# LIS on toys with intfloat/e5-base-v2"

# LIS on sports with intfloat/e5-base-v2"
