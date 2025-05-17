import pandas as pd
from src.utils.project_dirs import get_gen_dir_dataset, processed_data_dir, get_peruser_metric_dataset_modelname, get_dense_retrieval_index_dir, get_peruser_metric_dataset_modelname_encoder
from src.calcirmetrics_bm25_dataset_agno import load_data, verify_reviewer_ids, get_qrels
from ranx import Qrels, Run, evaluate
from typing import List, Dict
import os
from tqdm import tqdm
from collections import defaultdict
import argparse
from retriv import DenseRetriever
import numpy as np

# --- Dataset Configurations ---
# This dictionary centralizes all dataset-specific information
dataset_configs = {
    "amazon": {  # Base configuration for Amazon-like datasets
        "user_id_key": "reviewer_id",
        "item_id_key": "asin",
        "meta_columns": ['asin', 'title'],  # Expected columns for meta_corpus
        "nlang_cols": ['title'],
        "nlang_prefix_map": {'title': 'Title: '},
    },
    "movielens": {
        "user_id_key": "user_id",
        "item_id_key": "movie_id",
        "meta_columns": ['movie_id', 'title', 'genre'],  # Expected columns for meta_corpus
        "nlang_cols": ['title', 'genre'],
        "nlang_prefix_map": {'title': 'Title: ', 'genre': 'Genres: '},
    }
    # Add more categories/datasets here as needed
}

def get_rundict_from_dense_faster(genop: List[Dict], dr: DenseRetriever, num_return_sequences: int, user_id_key: str, item_id_key: str, batch_size: int = None):
    """
    Batched version of dense retrieval and formatting into a run dictionary.

    Args:
        genop: List of dictionaries, where each contains generated sequences and user IDs.
        dr: The DenseRetriever object.
        num_return_sequences: Number of generated sequences per user.
        user_id_key: Key to access user IDs in genop.
        item_id_key: Key to access item IDs.
        batch_size: Batch size for the dense retrieval.

    Returns:
        A dictionary formatted as {user_id: {item_id: score, ...}, ...}.
    """
    if batch_size is None:
        batch_size = num_return_sequences

    run_dict = defaultdict(dict)
    l = len(genop)
    all_queries = []
    # first get all the queries
    for i in range(l):
        reviewer_id = genop[i][user_id_key]  # Use user_id_key
        generated_seqs = genop[i]['generated_sequences']
        queries = [{'id': f'{reviewer_id}_{j}', 'text': generated_seqs[j]} for j in range(num_return_sequences)]
        all_queries.extend(queries)

    # batched search
    res = dr.bsearch(queries=all_queries, cutoff=1, batch_size=batch_size,show_progress=False)  # dictionary of reviewer_id_j -> {item_id: score} # Changed asin to item_id
    # first sort by reviewer_id
    for k, v in res.items():
        k = k.split('_')[0]  # the reviewer id
        item_id, score = next(iter(v.items()))  # Changed asin to item_id
        run_dict[k][item_id] = score  # Changed asin to item_id.  k here should be reviewer_id.  Was k before.
    return run_dict

def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int, at_k: int, config: Dict, batch_size: int = 128, metrics: List[str] = None):
    """
    Evaluates the dense retrieval performance.

    Args:
      genop: list of dicts, each dict contains
      retriever_filepath:
      num_return_sequences:
      at_k:
      config: dataset config
      batch_size:
      metrics:
    """
    if metrics is None:
        metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    qrels = Qrels(get_qrels(genop, config["user_id_key"], config["item_id_key"]))
    dr = DenseRetriever.load(index_name=retriever_filepath)
    run_dict = get_rundict_from_dense_faster(genop, dr, num_return_sequences, config["user_id_key"], config["item_id_key"],
                                            batch_size=batch_size)
    rundR = Run(run_dict)
    ans = evaluate(qrels, rundR, metrics)
    return qrels, rundR, ans

def create_dense_retrieval_index(meta_filepath: str, retriever_filepath: str, item_id_key: str,
                                 max_length: int = 256,
                                 encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                                 use_ann: bool = False,
                                 use_gpu: bool = True, batch_size: int = 128,
                                 show_progress: bool = False, config: Dict = None, item_dict: Dict = None):
    """
    Creates a dense retrieval index for the given meta corpus.

    Args:
        meta_filepath: Path to the meta corpus file (JSON lines).
        retriever_filepath: Path to save the dense retrieval index.
        item_id_key: Key to access item IDs in the meta corpus.
        max_length: Maximum length of the text to be embedded.
        encoder_name: Name of the sentence transformer model.
        use_ann: Whether to use an approximate nearest neighbor index.
        use_gpu: Whether to use GPU for index creation.
        batch_size: Batch size for processing.
        show_progress: Whether to display progress.
        config: the dataset config
        item_dict: the item id to nlang dict
    """
    dr = DenseRetriever(
        index_name=retriever_filepath,
        model=encoder_name,
        normalize=True,
        max_length=max_length,
        use_ann=use_ann
    )

    # Load the meta_corpus using the same function as the main script
    meta_corpus = pd.read_json(meta_filepath, orient='records', lines=True)
    meta_corpus = meta_corpus.astype(str) # Ensure every column is a string

     # Apply column renaming from config if specified
    if config is not None and 'meta_columns' in config and np.all((config['meta_columns']) == (meta_corpus.columns)):
        meta_corpus.columns = config['meta_columns']
    elif config is not None and 'meta_columns' in config and len(config['meta_columns']) != len(meta_corpus.columns):
        print(f"Warning: config['meta_columns'] length ({len(config['meta_columns'])}) does not match actual meta_corpus columns length ({len(meta_corpus.columns)}). "
              f"Proceeding without automatic column renaming based on config. Ensure your meta_corpus has the correct column names: {config['meta_columns']}")

    dr = dr.index_file(
        path=meta_filepath,
        embeddings_path=None,
        use_gpu=use_gpu,
        batch_size=batch_size,
        show_progress=show_progress,
        callback=lambda doc: {
            "id": doc[item_id_key],
            "text": item_dict[doc[item_id_key]],  # Use the dict for efficient lookup
        }
    )
    dr.save()


def get_metrics(meta_filepath: str, generated_filepath: str,
                retriever_filepath: str, num_sequences: int, at_k: int, dataset_name: str,
                config: Dict,  # Add the config dict
                max_length: int = 256, encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                use_ann: bool = False,
                use_gpu: bool = True, batch_size: int = 128, show_progress: bool = False,
                peruser_savepath: str = None, metrics: List[str] = None): # Added metrics
    """
    Main function to load data, create/load dense retrieval index, evaluate, and print results.
    Args:
        meta_filepath: Path to the meta corpus file.
        generated_filepath: Path to the generated sequences file.
        retriever_filepath: Path to the dense retrieval index.
        num_sequences: Number of generated sequences per user.
        at_k: Cutoff rank for evaluation metrics.
        dataset_name: Name of the dataset.
        config: dataset config
        max_length: Maximum length of the text to be embedded.
        encoder_name: Name of the sentence transformer model.
        use_ann: Whether to use an approximate nearest neighbor index.
        use_gpu: Whether to use GPU.
        batch_size: Batch size for processing.
        show_progress: Whether to show progress.
        peruser_savepath: Path to save per-user metrics.
        metrics:
    """
    print("Starting the evaluation process...")
    # Load data
    print("Loading data...")
    items_compact, genop = load_data(meta_filepath, generated_filepath, config)  # Use the agnostic load_data
    item_dict = items_compact.set_index(config["item_id_key"])['nlang'].to_dict()
    # Verify reviewer IDs
    print("Verifying reviewer IDs...")
    verify_reviewer_ids(genop, config["user_id_key"]) # Use the agnostic verify_reviewer_ids

    # make denser retrieval index if not already made
    if not os.path.exists(retriever_filepath):
        print("Making denser retrieval index...")
        create_dense_retrieval_index(meta_filepath, retriever_filepath, config["item_id_key"],
                                      max_length, encoder_name, use_ann, use_gpu, batch_size,
                                      show_progress, config=config, item_dict=item_dict)
    else:
        print("Dense retrieval index already exists...")
        dr = DenseRetriever.load(retriever_filepath)

    # Evaluate retrieval
    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences, at_k,
                                            config,  # Pass the config
                                            batch_size=batch_size, metrics=metrics) # Added metrics
    perusermetrics = rundR.scores
    df_metrics = pd.DataFrame(perusermetrics)
    if peruser_savepath:
      df_metrics.to_csv(peruser_savepath + ".csv", index=False)
      df_metrics.to_json(peruser_savepath + ".jsonl", orient="records")

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {dataset_name}")  # Changed from category to dataset_name
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print("Metrics:", ans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Specific dataset name (e.g., 'beauty', 'ml100k', 'sports').")
    parser.add_argument("--data_family", type=str, required=True,choices=["amazon", "movielens"],help="Family of the dataset (e.g., 'amazon', 'movielens').")
    parser.add_argument("--generated_file", type=str, required=True,help="The JSON file containing generated sequences (e.g., 'val_gen_op.json').")
    parser.add_argument("--split", type=str, required=True, help="The split to evaluate on (e.g., 'validation', 'test').")
    parser.add_argument("--short_model_name", type=str, required=True, help="The short model name (e.g., 'llama-1b').")
    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")
    parser.add_argument("--at_k", type=int, default=5, help="Number of return sequences to consider per reviewer.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for dense retrieval.")
    parser.add_argument("--metrics", nargs='+', default=["recall@5", "ndcg@5", "mrr"], help="Metrics to evaluate.")
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder name.")



    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    data_family = args.data_family.lower()

    current_config = None
    if data_family == "amazon":
        current_config = dataset_configs["amazon"].copy()
    elif data_family == "movielens":
        current_config = dataset_configs["movielens"].copy()
    else:
        raise ValueError(f"Unsupported data_family: '{data_family}'. Please define its configuration.")

    generated_filepath = str(get_gen_dir_dataset(dataset_name) / args.generated_file)
    meta_filepath = str(processed_data_dir(dataset_name) / 'meta_corpus.jsonl')
    retriever_filepath = str(get_dense_retrieval_index_dir(args.encoder_name) / f'{dataset_name}_index')
    at_k = args.at_k

    filename = (args.dataset_name + "_" + args.split + "_" + args.short_model_name)
    peruser_savepath = str(get_peruser_metric_dataset_modelname_encoder(args.dataset_name, args.short_model_name, args.encoder_name) / filename)

    get_metrics(meta_filepath=meta_filepath, generated_filepath=generated_filepath,
                retriever_filepath=retriever_filepath, num_sequences=args.num_sequences,
                at_k=at_k, dataset_name=dataset_name,
                config=current_config,
                batch_size=args.batch_size, peruser_savepath=peruser_savepath, metrics=args.metrics, encoder_name=args.encoder_name)

# Toys, llama-8b, TEST with all-MiniLM-L6-v2
# Metrics: {'recall@5': 0.07960814642949214, 'ndcg@5': 0.05264008851607378, 'mrr': 0.043805963736358165}

# ML100k, llama-8b, TEST with all-MiniLM-L6-v2
# Metrics: {'recall@5': 0.060445387062566275, 'ndcg@5': 0.03594454348807995, 'mrr': 0.027942735949098622}