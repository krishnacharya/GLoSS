"""
Dense retrieval with FAISS + sentence-transformers.
Stage 1: Top-1 per query (mirrors calcirmetrics_bm25_dataset_agno).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate
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
    get_unique_sorted_items,
    dataset_configs,
)


def build_or_load_faiss_index(
    items_compact: pd.DataFrame,
    item_id_key: str,
    index_path: str,
    asins_path: str,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    show_progress: bool = True,
) -> tuple[faiss.Index, List[str], SentenceTransformer]:
    """
    Build FAISS index from catalog texts, or load if exists.

    Returns:
        (faiss_index, catalog_asins, model)
    """
    catalog_asins = items_compact[item_id_key].tolist()
    catalog_texts = items_compact["nlang"].tolist()

    if os.path.exists(index_path) and os.path.exists(asins_path):
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        with open(asins_path, "r") as f:
            catalog_asins = json.load(f)
        model = SentenceTransformer(encoder_name)
        return index, catalog_asins, model

    print(f"Building FAISS index from catalog...")
    model = SentenceTransformer(encoder_name)
    catalog_embeddings = model.encode(
        catalog_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    catalog_embeddings = catalog_embeddings.astype(np.float32)

    embed_dim = catalog_embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(catalog_embeddings)

    index_dir = os.path.dirname(index_path)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(asins_path, "w") as f:
        json.dump(catalog_asins, f)
    print(f"Saved FAISS index to {index_path}, catalog ASINs to {asins_path}")

    return index, catalog_asins, model


def get_rundict_from_dense(
    genop: List[Dict],
    faiss_index: faiss.Index,
    catalog_asins: List[str],
    model: SentenceTransformer,
    num_return_sequences: int,
    user_id_key: str,
    item_id_key: str,
    batch_size: int = 512,
) -> Dict[str, Dict[str, float]]:
    """
    Dense retrieval: top-1 per query. Mirrors get_rundict from calcirmetrics_bm25_dataset_agno.
    """
    num_users = len(genop)
    if num_return_sequences != len(genop[0]["generated_sequences"]):
        raise ValueError(
            f"num_return_sequences {num_return_sequences} "
            f"does not match {len(genop[0]['generated_sequences'])}"
        )

    queries_flat = [seq for row in genop for seq in row["generated_sequences"]]
    query_embeddings = model.encode(
        queries_flat,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    query_embeddings = query_embeddings.astype(np.float32)

    scores, indices = faiss_index.search(query_embeddings, 1)
    indices = indices.reshape(num_users, num_return_sequences)
    scores = scores.reshape(num_users, num_return_sequences)

    run_dict = {}
    for i in range(num_users):
        reviewer_id = genop[i][user_id_key]
        item_indices = indices[i]
        item_scores = scores[i].tolist()
        items = [catalog_asins[idx] for idx in item_indices if 0 <= idx < len(catalog_asins)]
        scores_for_items = [
            s for idx, s in zip(item_indices, item_scores)
            if 0 <= idx < len(catalog_asins)
        ]
        run_dict[reviewer_id] = get_unique_sorted_items(items, scores_for_items)
    return run_dict


def evaluate_retrieval(
    genop: List[Dict],
    faiss_index: faiss.Index,
    catalog_asins: List[str],
    model: SentenceTransformer,
    num_return_sequences: int,
    at_k: int,
    config: Dict,
    batch_size: int = 512,
    metrics: List[str] = None,
):
    if metrics is None:
        metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    qrels = Qrels(get_qrels(genop, config["user_id_key"], config["item_id_key"]))
    run_dict = get_rundict_from_dense(
        genop, faiss_index, catalog_asins, model,
        num_return_sequences, config["user_id_key"], config["item_id_key"],
        batch_size=batch_size,
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
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    show_progress: bool = True,
    peruser_savepath: str = None,
    metrics: List[str] = None,
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

    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(
        genop, faiss_index, catalog_asins, model,
        num_sequences, at_k, config,
        batch_size=batch_size,
        metrics=metrics,
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
    print("Metrics:", ans)
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval (FAISS) performance based on generated sequences."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="e.g., 'beauty', 'sports'")
    parser.add_argument("--data_family", type=str, required=True, choices=["amazon"])
    parser.add_argument("--generated_file", type=str, required=True, help="JSON file with generated sequences")
    parser.add_argument("--split", type=str, required=True, help="e.g., 'validation', 'test'")
    parser.add_argument("--short_model_name", type=str, required=True, help="e.g., 'llama-1b'")
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--at_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--metrics", nargs="+", default=["recall@5", "ndcg@5", "mrr"])
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

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
    encoder_for_peruser = f"faiss_{args.encoder_name.replace('/', '_')}"
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
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        peruser_savepath=peruser_savepath,
        metrics=args.metrics,
    )
