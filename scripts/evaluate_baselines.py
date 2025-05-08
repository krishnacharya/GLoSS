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

def load_datasets(dataset_path: str) -> DatasetDict:
    """Loads a Hugging Face DatasetDict from disk."""
    return load_from_disk(dataset_path)


def prepare_metadata_for_retrieval(meta_corpus_path: str) -> (pd.DataFrame, Dict[str, str]):
    """Loads and prepares metadata for BM25 retrieval."""
    meta_corpus = pd.read_json(meta_corpus_path, orient='records', lines=True)
    meta_corpus.columns = ['asin', 'Title']
    asins_compact = meta_corpus[['asin']].copy()
    asins_compact['nlang'] = "Title: " + meta_corpus['Title']
    asin_dict = asins_compact.set_index('asin')['nlang'].to_dict()
    return asins_compact, asin_dict


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


def get_qrels(dataset: Dataset) -> Dict[str, Dict[str, int]]:
    """Generates ground truth (qrels) from the dataset."""
    qrels_dict = {}
    for row in dataset:
        reviewer_id = row['reviewer_id']
        asin = row['asin']
        if reviewer_id not in qrels_dict:
            qrels_dict[reviewer_id] = {}
        qrels_dict[reviewer_id][asin] = 1
    return qrels_dict


def randrecs_fromcorpus(meta_corpus: pd.DataFrame, dataset: Dataset, k: int = 5, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Generates random recommendations from the item corpus."""
    np.random.seed(seed)
    run_dict = {}
    all_asins = meta_corpus['asin'].unique().tolist()
    for row in dataset:
        reviewer_id = row['reviewer_id']
        random_asins = np.random.choice(all_asins, size=k, replace=False)
        run_dict[reviewer_id] = {asin: 1.0 for asin in random_asins}
    return run_dict


def randrecs_popweighted(meta_corpus: pd.DataFrame, dataset: Dataset, item_popularity: pd.Series, k: int = 5, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Generates popularity-weighted random recommendations."""
    np.random.seed(seed)
    run_dict = {}
    all_asins = item_popularity.index.tolist()
    probabilities = item_popularity.values / item_popularity.sum()
    for row in dataset:
        reviewer_id = row['reviewer_id']
        sampled_asins = np.random.choice(all_asins, size=k, replace=False, p=probabilities)
        scores = {asin: item_popularity.get(asin, 0) for asin in sampled_asins}
        run_dict[reviewer_id] = scores
    return run_dict


def repeat_last_seen(dataset: Dataset, item_popularity: pd.Series, k: int = 5, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Recommends the last k seen items with recency-based scores and backfills with popularity."""
    np.random.seed(seed)
    run_dict = {}
    popular_asins = item_popularity.index.tolist()
    popularity_probs = item_popularity.values / item_popularity.sum()

    for row in dataset:
        reviewer_id = row['reviewer_id']
        seen_asins = row['seen_asins']
        num_seen = len(seen_asins)
        recommendations = {}
        for i in range(min(k, num_seen)):
            asin = seen_asins[num_seen - 1 - i]
            recommendations[asin] = k - i
        if len(recommendations) < k:
            backfill_count = k - len(recommendations)
            eligible_backfill_asins = [asin for asin in popular_asins if asin not in recommendations]
            if eligible_backfill_asins:
                eligible_probs = item_popularity[eligible_backfill_asins].values / item_popularity[eligible_backfill_asins].sum()
                backfilled_asins = np.random.choice(eligible_backfill_asins, size=backfill_count, replace=False, p=eligible_probs)
                for asin in backfilled_asins:
                    recommendations[asin] = 0.1
        run_dict[reviewer_id] = recommendations
    return run_dict


def textbased_lastsimilar(dataset: Dataset, retriever: bm25s.BM25, asin_dict: Dict[str, str], asins_compact: pd.DataFrame, k: int = 5) -> Dict[str, Dict[str, float]]:
    """Generates recommendations based on text similarity of the last seen item using BM25."""
    l = len(dataset)
    queries_flat = [asin_dict[seen[-1]] for seen in dataset['seen_asins']]
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en")
    res, scores = retriever.retrieve(query_tokens, k=k)
    run_dict = {}
    for i in tqdm(range(l), desc="Generating Recommendations"):
        reviewer_id = dataset['reviewer_id'][i]
        asin_indices = res[i]
        asin_scores = scores[i]
        asins = asins_compact.iloc[asin_indices]['asin'].tolist()
        run_dict[reviewer_id] = {asin: score for asin, score in zip(asins, asin_scores)}
    return run_dict


def evaluate_model(qrels: Qrels, run: Run, metrics: List[str], model_name: str):
    """Evaluates a recommendation model and prints the results."""
    print(f"\nEvaluating {model_name}:")
    print("-" * 30)
    results = evaluate(qrels, run, metrics)
    print(results)
    print("-" * 30)
    return results


def main(dataset_path: str, meta_corpus_path: str, bm25_index_name: str = "amznbeauty2014_index", \
        k: int = 5, metrics: List[str] = ["recall@5", "ndcg@5", "mrr"], dataset_split: str = "test"):
    """Main function to load data, build baselines, and evaluate them."""
    # Load datasets
    dataset = load_datasets(dataset_path)

    # Prepare metadata for BM25
    asins_compact, asin_dict = prepare_metadata_for_retrieval(meta_corpus_path)
    corpus_list = asins_compact['nlang'].tolist()

    # Build or load BM25 retriever
    bm25_index_path = str(get_bm25_indexes_dir() / bm25_index_name)
    retriever = build_bm25_retriever(corpus_list, bm25_index_path)

    # Prepare ground truth
    qrels_test = Qrels(get_qrels(dataset[dataset_split]))
    print(f"\nQrels ({dataset_split}):")
    print("-" * 30)

    # # Load item popularity
    # core5 = pd.read_json(dataset_path.replace("df_withdup", "reviews_Beauty_5.json"), lines=True) # Load original core5 for popularity
    # item_pop = core5['asin'].value_counts()

    # # Evaluate Random Recommendations
    # run_test_rand = Run(randrecs_fromcorpus(asins_compact[['asin']], dataset[dataset_split], k=k))
    # evaluate_model(qrels_test, run_test_rand, metrics, "Random Recommendations")

    # # Evaluate Popularity-Weighted Recommendations
    # run_test_popweighted = Run(randrecs_popweighted(asins_compact[['asin']], dataset[dataset_split], item_pop, k=k))
    # evaluate_model(qrels_test, run_test_popweighted, metrics, "Popularity-Weighted Recommendations")

    # # Evaluate Repeat Last Seen Recommendations
    # run_test_repeat_last = Run(repeat_last_seen(dataset[dataset_split], item_pop, k=k))
    # evaluate_model(qrels_test, run_test_repeat_last, metrics, "Repeat Last Seen Recommendations")

    # Evaluate Text-Based Last Similar Recommendations
    run_test_textbased_lastsimilar = Run(textbased_lastsimilar(dataset[dataset_split], retriever, asin_dict, asins_compact, k=k))
    evaluate_model(qrels_test, run_test_textbased_lastsimilar, metrics, "Text-Based Last Similar Recommendations")


if __name__ == "__main__":
    hf_dir = get_hfdata_dir()
    processed_dir = processed_data_dir('beauty2014')
    dataset_path_withdup = str(hf_dir / 'beauty2014')
    meta_corpus_path = str(processed_dir / 'meta_corpus.json')
    main(dataset_path_withdup, meta_corpus_path, bm25_index_name="amznbeauty2014_index", dataset_split="test")

# on test LIR
# {'recall@5': np.float64(0.04333050127442651), 'ndcg@5': np.float64(0.023179244762936268), 'mrr': np.float64(0.016472894215147044)}

# on validation LIR
# {'recall@5': np.float64(0.03577817531305903), 'ndcg@5': np.float64(0.01862776408186558), 'mrr': np.float64(0.012939773404889682)}
