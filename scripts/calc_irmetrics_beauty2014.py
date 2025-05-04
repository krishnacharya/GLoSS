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

def get_qrels(genop: List[Dict]) -> Dict[str, Dict[str, int]]:
    '''
    Returns: dict[reviewer_id] = {asin1: 1, asin2: 1, ...}
    '''
    qrels_dict = defaultdict(dict)
    for row in genop:
        reviewer = row['reviewer_id']
        asin = row['asin']
        qrels_dict[reviewer][asin] = 1
    return dict(qrels_dict)

def get_unique_sorted_asins(asins: List[str], scores: List[float]) -> Dict[str, float]:
    """Helper: return unique asins sorted by score."""
    seen = set()
    asin_score_pairs = []
    for asin, score in sorted(zip(asins, scores), key=lambda x: -x[1]):
        if asin not in seen:
            asin_score_pairs.append((asin, score))
            seen.add(asin)
    return dict(asin_score_pairs)

def get_rundict(genop: List[Dict], retriever: bm25s.BM25, num_return_sequences: int, asins_compact: pd.DataFrame) -> Dict[str, Dict[str, float]]:
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
        reviewer_id = genop[i]['reviewer_id']
        asin_indices = res[i]
        asin_scores = scores[i]
        asins = asins_compact.iloc[asin_indices]['asin'].tolist()
        run_dict[reviewer_id] = get_unique_sorted_asins(asins, asin_scores)

    return run_dict

def verify_reviewer_ids(valgen: List[Dict]) -> None:
    """Verifies the number and uniqueness of reviewer IDs."""
    reviewers = [row['reviewer_id'] for row in valgen]
    print(f"Number of reviewers: {len(reviewers)}")
    print(f"Number of unique reviewers: {len(set(reviewers))}")

def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int, asins_compact: pd.DataFrame,
                       at_k: int):
    """Evaluates the retrieval performance."""
    qrels = Qrels(get_qrels(genop))
    retriever = bm25s.BM25.load(retriever_filepath, load_corpus=False)
    run_dict = get_rundict(genop, retriever, num_return_sequences, asins_compact)
    rundR = Run(run_dict)
    metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    ans = evaluate(qrels, rundR, metrics)
    return qrels, rundR, ans


def load_data(meta_filepath: str, generated_filepath: str) -> tuple[pd.DataFrame, List[Dict]]:
    """Loads the meta data and the generated sequences."""
    meta_corpus = pd.read_json(meta_filepath, orient='records', lines=True)
    meta_corpus.columns = ['asin', 'Title']
    asins_compact = meta_corpus[['asin']].copy()
    asins_compact['nlang'] = "Title: " + meta_corpus['Title']
    asin_dict = asins_compact.set_index('asin')['nlang'].to_dict() # asin to serialized natural language string
    with open(generated_filepath, "r") as f:
        genop = json.load(f)
    print(f"Loaded generated data: type={type(genop)}, first element length={len(genop[0]) if genop else 0}")
    return asins_compact, genop

def main(meta_filepath: str, generated_filepath: str, retriever_filepath: str, num_sequences: int, at_k: int):
    """Main function to load data, evaluate retrieval, and print results."""
    print("Starting the evaluation process...")

    # Load data
    print("Loading data...")
    asins_compact, genop = load_data(meta_filepath, generated_filepath)

    # Verify reviewer IDs
    print("Verifying reviewer IDs...")
    verify_reviewer_ids(genop)

    # Evaluate retrieval
    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences, asins_compact, at_k)

    print("\n--- Evaluation Summary ---")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print("Metrics:", ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences.")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to the meta corpus JSON file.")
    parser.add_argument("--generated_file", type=str, required=True, help="Path to the JSON file containing generated sequences.")
    parser.add_argument("--retriever_index", type=str, required=True, help="Path to the BM25 retriever index file.")
    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")

    args = parser.parse_args()
    generated_filepath = str(get_gen_dir_dataset('beauty2014') / args.generated_file)
    meta_filepath = str(processed_data_dir('beauty2014') / args.meta_file)
    retriever_filepath = str(get_bm25_indexes_dir() / args.retriever_index)
    at_k = args.num_sequences
    main(meta_filepath, generated_filepath, retriever_filepath, args.num_sequences, at_k)


# on test LLama1BftBeauty2014 with beam search
# Metrics: {'recall@5': np.float64(0.04534275365559182), 'ndcg@5': np.float64(0.026878945255920418), 'mrr': np.float64(0.020865566635364964)}

# on validation LLama1BftBeauty2014  with beam search
# Metrics: {'recall@5': np.float64(0.05366726296958855), 'ndcg@5': np.float64(0.03218498256886583), 'mrr': np.float64(0.025163983303518187)}

# on test LLama1BftBeauty2014 with sampling 0.5

# onn validation LLama1BftBeauty2014 with sampling, temp 0.5
# Metrics: {'recall@5': np.float64(0.01073345259391771), 'ndcg@5': np.float64(0.006946322436661294), 'mrr': np.float64(0.005724508050089446)}

# on validation llama1b sampling, temp = 0.2
# Metrics: {'recall@5': np.float64(0.011627906976744186), 'ndcg@5': np.float64(0.008841746091621304), 'mrr': np.float64(0.007901013714967202)}

# on validation llama1b sampling, temp = 0.1
# Metrics: {'recall@5': np.float64(0.006261180679785331), 'ndcg@5': np.float64(0.005600947680807617), 'mrr': np.float64(0.005366726296958855)}
# -----------------------------
# on test Qwen3-0.6B-ftBeauty2014


# on validation Qwen3-0.6B-ftBeauty2014
# Metrics: {'recall@5': np.float64(0.01073345259391771), 'ndcg@5': np.float64(0.010286225402504472), 'mrr': np.float64(0.010137149672033392)}

# on test Gemma3-1B-ftBeauty2014
# -----------------------------

# on val SmolLM160M-ftBeauty2014 with beam search
# Metrics: {'recall@5': np.float64(0.0017889087656529517), 'ndcg@5': np.float64(0.0014587922661640944), 'mrr': np.float64(0.0013416815742397137)}

# on val SmolLM160M-ftBeauty2014 with sampling temp 0.5
# Metrics: {'recall@5': np.float64(0.014311270125223614), 'ndcg@5': np.float64(0.009315474775102668), 'mrr': np.float64(0.007692307692307692)}