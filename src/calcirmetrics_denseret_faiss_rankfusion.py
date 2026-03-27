"""
Dense retrieval with FAISS + sentence-transformers.
Stage 2: Top-k per query + fusion (mirrors calcirmetrics_bm25_dataset_agno_rankfusion).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate, fuse
from typing import List, Dict

from src.utils.project_dirs import (
    get_gen_dir_dataset,
    processed_data_dir,
    get_dense_retrieval_index_dir,
    get_peruser_metric_dataset_modelname_encoder,
)
from src.calcirmetrics_bm25_dataset_agno import (
    load_data,
    get_qrels,
    verify_reviewer_ids,
    dataset_configs,
)
from src.calcirmetrics_denseret_faiss import build_or_load_faiss_index


def retrieve_with_topk_dense(
    genop: List[Dict],
    faiss_index: faiss.Index,
    catalog_asins: List[str],
    model: SentenceTransformer,
    num_return_sequences: int,
    user_id_key: str,
    item_id_key: str,
    top_k: int,
    batch_size: int = 512,
) -> Dict[str, List[Dict]]:
    """
    Retrieves topK items for each generated query via dense retrieval. Groups by user.
    Mirrors retrieve_with_topk from calcirmetrics_bm25_dataset_agno_rankfusion.

    Returns:
        Dict mapping user_id to list of retrieval results, where each result contains:
        - 'query_idx': index of the generated query (0 to num_return_sequences-1)
        - 'items': list of item IDs (topK)
        - 'scores': list of dense similarity scores (topK)
        - 'item_indices': list of item indices in catalog
    """
    num_users = len(genop)
    if num_return_sequences != len(genop[0]["generated_sequences"]):
        raise ValueError(
            f"num_return_sequences {num_return_sequences} "
            f"does not match the number of generated sequences {len(genop[0]['generated_sequences'])}"
        )

    queries_flat = [seq for row in genop for seq in row["generated_sequences"]]
    query_embeddings = model.encode(
        queries_flat,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    query_embeddings = query_embeddings.astype(np.float32)

    scores, indices = faiss_index.search(query_embeddings, top_k)
    print(f"Retrieved results shape: {indices.shape}, scores shape: {scores.shape}")

    indices_reshaped = indices.reshape((num_users, num_return_sequences, top_k))
    scores_reshaped = scores.reshape((num_users, num_return_sequences, top_k))

    user_results = {}
    for i in range(num_users):
        reviewer_id = genop[i][user_id_key]
        user_results[reviewer_id] = []

        for query_idx in range(num_return_sequences):
            item_indices = indices_reshaped[i, query_idx]
            item_scores = scores_reshaped[i, query_idx]
            items = [
                catalog_asins[idx]
                for idx in item_indices
                if 0 <= idx < len(catalog_asins)
            ]
            scores_list = [
                float(s)
                for idx, s in zip(item_indices, item_scores)
                if 0 <= idx < len(catalog_asins)
            ]
            user_results[reviewer_id].append({
                "query_idx": query_idx,
                "items": items,
                "scores": scores_list,
                "item_indices": [int(x) for x in item_indices if 0 <= x < len(catalog_asins)],
            })

    return user_results


def automated_fusion(
    user_results_dict: Dict[str, List[Dict]], method: str = "sum", norm: str = "min-max"
) -> Dict[str, Dict[str, float]]:
    """
    Use ranx.fuse() to aggregate scores across queries for all users.
    Same logic as calcirmetrics_bm25_dataset_agno_rankfusion.automated_fusion.
    """
    if not user_results_dict:
        return {}

    first_user_results = next(iter(user_results_dict.values()))
    num_queries = len(first_user_results)

    query_runs = []
    for query_idx in range(num_queries):
        query_run_dict = {}
        for user_id, user_results in user_results_dict.items():
            query_result = None
            for result in user_results:
                if result.get("query_idx") == query_idx:
                    query_result = result
                    break
            if query_result is None:
                continue
            items = query_result["items"]
            scores = query_result["scores"]
            user_item_scores = {item: score for item, score in zip(items, scores)}
            query_run_dict[user_id] = user_item_scores
        query_run = Run(query_run_dict)
        query_runs.append(query_run)

    fused_run = fuse(runs=query_runs, norm=norm, method=method)
    return fused_run.run


def get_rundict_with_fusion(
    genop: List[Dict],
    faiss_index: faiss.Index,
    catalog_asins: List[str],
    model: SentenceTransformer,
    num_return_sequences: int,
    user_id_key: str,
    item_id_key: str,
    top_k: int,
    method: str = "sum",
    norm: str = "min-max",
    batch_size: int = 512,
) -> Dict[str, Dict[str, float]]:
    """Retrieves topK per query and fuses across the 5 queries per user."""
    user_results_dict = retrieve_with_topk_dense(
        genop, faiss_index, catalog_asins, model,
        num_return_sequences, user_id_key, item_id_key, top_k,
        batch_size=batch_size,
    )
    return automated_fusion(user_results_dict, method=method, norm=norm)


def evaluate_retrieval(
    genop: List[Dict],
    faiss_index: faiss.Index,
    catalog_asins: List[str],
    model: SentenceTransformer,
    num_return_sequences: int,
    at_k: int,
    config: Dict,
    top_k: int = 10,
    method: str = "sum",
    norm: str = "min-max",
    batch_size: int = 512,
    metrics: List[str] = None,
):
    if metrics is None:
        metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    qrels = Qrels(get_qrels(genop, config["user_id_key"], config["item_id_key"]))
    run_dict = get_rundict_with_fusion(
        genop, faiss_index, catalog_asins, model,
        num_return_sequences, config["user_id_key"], config["item_id_key"],
        top_k, method=method, norm=norm, batch_size=batch_size,
    )
    rundR = Run(run_dict)
    ans = evaluate(qrels, rundR, metrics)
    return qrels, rundR, ans


def get_metrics(
    meta_filepath: str,
    generated_filepath: str,
    index_path: str,
    asins_path: str,
    num_sequences: int,
    at_k: int,
    dataset_name: str,
    config: Dict,
    top_k: int = 10,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    show_progress: bool = True,
    peruser_savepath: str = None,
    metrics: List[str] = None,
    method: str = "sum",
    norm: str = "min-max",
):
    print("Starting the evaluation process...")
    print("Loading data...")
    items_compact, genop = load_data(meta_filepath, generated_filepath, config)
    verify_reviewer_ids(genop, config["user_id_key"])

    print("Building or loading FAISS index...")
    faiss_index, catalog_asins, model = build_or_load_faiss_index(
        items_compact, config["item_id_key"],
        index_path, asins_path,
        encoder_name=encoder_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    print("Evaluating retrieval performance with rank fusion...")
    qrels, rundR, ans = evaluate_retrieval(
        genop, faiss_index, catalog_asins, model,
        num_sequences, at_k, config,
        top_k=top_k, method=method, norm=norm, batch_size=batch_size, metrics=metrics,
    )

    perusermetrics = rundR.scores
    df_metrics_list = []
    for metric_name, scores_dict in perusermetrics.items():
        df_metric = pd.DataFrame.from_dict(scores_dict, orient="index", columns=[metric_name])
        df_metrics_list.append(df_metric)

    if df_metrics_list:
        df_metrics = pd.concat(df_metrics_list, axis=1, join="outer")
        user_id_column_name = config.get("user_id_key", "user_id")
        df_metrics.index.name = user_id_column_name
        df_metrics = df_metrics.reset_index()
    else:
        user_id_column_name = config.get("user_id_key", "user_id")
        df_metrics = pd.DataFrame(columns=[user_id_column_name])

    if peruser_savepath and not df_metrics.empty:
        df_metrics.to_json(peruser_savepath + ".jsonl", orient="records", lines=True)
        print(f"Per-user metrics saved to {peruser_savepath}.jsonl")

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {dataset_name}")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"FAISS index: {index_path}")
    print(f"Number of return sequences: {num_sequences}")
    print(f"Top-K retrieval depth: {top_k}")
    print(f"Fusion method: {method}")
    print("Metrics:", ans)
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval (FAISS) with rank fusion based on generated sequences."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="e.g., 'beauty', 'sports'")
    parser.add_argument("--data_family", type=str, required=True, choices=["amazon"])
    parser.add_argument("--generated_file", type=str, required=True, help="JSON file with generated sequences")
    parser.add_argument("--split", type=str, required=True, help="e.g., 'validation', 'test'")
    parser.add_argument("--short_model_name", type=str, required=True, help="e.g., 'llama-1b'")
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--at_k", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10, help="Top items to retrieve per query for fusion")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--metrics", nargs="+", default=["recall@5", "ndcg@5", "mrr"])
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="sum",
        choices=["min", "max", "med", "sum", "anz", "mnz", "rrf", "isr", "bordafuse", "condorcet"],
        help="Fusion method. See https://amenra.github.io/ranx/fusion/",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name.lower()
    data_family = args.data_family.lower()

    current_config = dataset_configs["amazon"].copy() if data_family == "amazon" else None
    if current_config is None:
        raise ValueError(f"Unsupported data_family: '{data_family}'")

    generated_filepath = str(get_gen_dir_dataset(dataset_name) / args.generated_file)
    meta_filepath = str(processed_data_dir(dataset_name) / "meta_corpus.jsonl")
    if not os.path.exists(meta_filepath):
        meta_filepath = str(processed_data_dir(dataset_name) / "meta_corpus.json")
    index_dir = get_dense_retrieval_index_dir(args.encoder_name)
    index_path = str(index_dir / f"{dataset_name}_faiss.index")
    asins_path = str(index_dir / f"{dataset_name}_faiss_asins.json")

    filename = f"{dataset_name}_{args.split}_{args.short_model_name}"
    encoder_for_peruser = f"faiss_{args.encoder_name.replace('/', '_')}_fused_{args.fusion_method}"
    peruser_savepath = str(
        get_peruser_metric_dataset_modelname_encoder(
            dataset_name, args.short_model_name, encoder_for_peruser
        ) / filename
    )

    get_metrics(
        meta_filepath=meta_filepath,
        generated_filepath=generated_filepath,
        index_path=index_path,
        asins_path=asins_path,
        num_sequences=args.num_sequences,
        at_k=args.at_k,
        dataset_name=dataset_name,
        config=current_config,
        top_k=args.top_k,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        peruser_savepath=peruser_savepath,
        metrics=args.metrics,
        method=args.fusion_method,
    )
