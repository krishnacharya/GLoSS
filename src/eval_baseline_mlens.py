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
from src.utils.project_dirs import processed_data_dir, get_hfdata_dir, get_bm25_indexes_dir
import argparse
import os
from src.evaluate_baselines import build_bm25_retriever, evaluate_model

def prepare_metadata_for_retrieval_mlfamily(meta_corpus_path: str):
    '''
        columns are: movie_id, title, genre
    '''
    meta_corpus = pd.read_json(meta_corpus_path, orient='records', lines=True)
    assert meta_corpus.columns.tolist() == ['movie_id', 'title', 'genre']
    meta_corpus = meta_corpus.astype(str)
    movies_compact = meta_corpus[['movie_id']].copy()
    movies_compact['nlang'] = "Title: " + meta_corpus['title'] + ", Genres: " + meta_corpus['genre']
    movieid_to_nlang = movies_compact.set_index('movie_id')['nlang'].to_dict()

    return movies_compact, movieid_to_nlang


def get_qrels_mlfamily(dataset: Dataset) -> Dict[str, Dict[str, int]]:
    '''
        Generates ground truth (qrels) from the dataset.
    '''
    qrels_dict = {}
    for row in dataset:
        reviewer_id = row['user_id']
        movie_id = row['movie_id']
        if reviewer_id not in qrels_dict:
            qrels_dict[reviewer_id] = {}
        qrels_dict[reviewer_id][movie_id] = 1
    return qrels_dict


def textbased_lastsimilar(dataset: Dataset, retriever: bm25s.BM25, 
                          movieid_to_nlang: Dict[str, str], movies_compact: pd.DataFrame, k: int = 5) -> Dict[str, Dict[str, float]]:
    """Generates recommendations based on text similarity of the last seen item using BM25."""
    l = len(dataset)
    queries_flat = [movieid_to_nlang[seen[-1]] for seen in dataset['seen_movie_ids']]
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en")
    res, scores = retriever.retrieve(query_tokens, k=k)
    run_dict = {}
    for i in tqdm(range(l), desc="Generating Recommendations"):
        reviewer_id = dataset['user_id'][i]
        movie_indices = res[i]
        movie_scores = scores[i]
        movies = movies_compact.iloc[movie_indices]['movie_id'].tolist()
        run_dict[reviewer_id] = {movie: score for movie, score in zip(movies, movie_scores)}
    return run_dict

def main(dataset_path: str, meta_corpus_path: str, bm25_index_name: str, \
        k: int = 5, metrics: List[str] = ["recall@5", "ndcg@5", "mrr"], dataset_split: str = "test"):
    dataset = load_from_disk(dataset_path)
    movies_compact, movieid_to_nlang = prepare_metadata_for_retrieval_mlfamily(meta_corpus_path)
    
    corpus_list = movies_compact['nlang'].tolist()

    # Build or load BM25 retriever
    bm25_index_path = str(get_bm25_indexes_dir() / bm25_index_name)
    retriever = build_bm25_retriever(corpus_list, bm25_index_path)
    

    # Prepare ground truth
    qrels_test = Qrels(get_qrels_mlfamily(dataset[dataset_split]))
    print(f"\nQrels ({dataset_split}):")
    print("-" * 30)

    # Evaluate Text-Based Last Similar Recommendations
    run_test_textbased_lastsimilar = Run(textbased_lastsimilar(dataset[dataset_split], retriever, movieid_to_nlang, movies_compact, k=k))
    evaluate_model(qrels_test, run_test_textbased_lastsimilar, metrics, "Text-Based Last Similar Recommendations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation models on a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="The path to the Hugging Face dataset.")
    parser.add_argument("--bm25_index_name", type=str, required=True, help="The name of the BM25 index.")
    parser.add_argument("--k", type=int, default=5, help="The value of k for evaluation metrics (e.g., recall@k).")
    parser.add_argument("--metrics", nargs='+', default=["recall@5", "ndcg@5", "mrr"], help="List of evaluation metrics to use.")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate (e.g., 'train', 'validation', 'test').")

    args = parser.parse_args()

    hf_dir = get_hfdata_dir()
    processed_dir = processed_data_dir(f"{args.dataset}")
    bm25_index_name = f"{args.dataset}_index"
    dataset_path = str(hf_dir / args.dataset)
    meta_corpus_path = str(processed_dir / 'meta_corpus.json')

    main(dataset_path, meta_corpus_path, args.bm25_index_name, args.k, args.metrics, args.split)

# ML100k test results with LIR
# {'recall@5': 0.021208907741251327, 'ndcg@5': 0.01158073939771433, 'mrr': 0.008377518557794273}