
from retriv import DenseRetriever
import pandas as pd
from src.utils.project_dirs import get_gen_dir_dataset, processed_data_dir, get_bm25_indexes_dir,get_minilm_index_dir
from src.calcirmetrics_bm25 import load_data, verify_reviewer_ids, get_qrels
from ranx import Qrels, Run, evaluate
from typing import List, Dict
import os
from tqdm import tqdm
from collections import defaultdict

def get_rundict_from_dense_retriever(genop: List[Dict], dr: DenseRetriever, num_return_sequences: int, asins_compact: pd.DataFrame):
    run_dict = defaultdict(dict)
    l = len(genop)

    if num_return_sequences != len(genop[0]['generated_sequences']):
        raise ValueError(f"num_return_sequences {num_return_sequences} "
                         f"does not match the number of generated sequences {len(genop[0]['generated_sequences'])}")

    # flatten all generated queries
    for i in tqdm(range(l), desc="Processing reviews"):
        reviewer_id = genop[i]['reviewer_id']
        generated_seqs = genop[i]['generated_sequences']
        queries = [{'id':f'{reviewer_id}_{j}', 'text':generated_seqs[j]} for j in range(num_return_sequences)] # hacky way because retriv expects each query id to be unique
        res = dr.msearch(queries=queries, cutoff=1, batch_size=num_return_sequences) # all for reviewer_id, returns a dictionary top asin text for each of the num_return_sequences genrated text {'rid_0': {'B00ECUPDYC': 0.66617227}, 'rid_1': {'B000052YM0': 0.72061837}}
        for k, v in res.items():
            asin, score = next(iter(v.items()))
            run_dict[reviewer_id][asin] = score
    return run_dict

def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int, asins_compact: pd.DataFrame,
                       at_k: int):
    '''
        genop: list of dicts, each dict contains 
    '''
    # qrels = Qrels(get_qrels(genop))
    # dr = DenseRetriever.load(index_name = retriever_filepath)
    # run_dict = get_rundict(genop, retriever, num_return_sequences, asins_compact)
    # rundR = Run(run_dict)
    # metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    # ans = evaluate(qrels, rundR, metrics)
    # return qrels, rundR, ans


def create_dense_retrieval_index(meta_filepath: str, retriever_filepath: str, max_length:int=256,
                                 model_name:str='sentence-transformers/all-MiniLM-L6-v2', use_ann:bool=False,
                                 use_gpu:bool=True, batch_size:int=128, show_progress:bool=True):
    '''
    Creates a dense retrieval index for the given meta asin corpus at the retriever_filepath.
    https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md

    If use_ann is True, uses an approximate nearest neighbor index, recommended not to use if less than 20k items
    use_gpu: bool, default True, use GPU for index creation
    '''
    dr = DenseRetriever(
        index_name = retriever_filepath,
        model = model_name,
        normalize=True,
        max_length=max_length, 
        use_ann=use_ann
    )
    dr = dr.index_file(path=str(meta_filepath/'meta_corpus.jsonl'), # has to be json lines, each seperated by new line
            embeddings_path=None,
            use_gpu=use_gpu,           
            batch_size=batch_size,            
            show_progress=show_progress,        
            callback=lambda doc: {
                "id": doc["asin"],
                "text": "Title: " + doc['title'],
            }
        )
    dr.save()

def get_metrics(meta_filepath: str, generated_filepath: str, \
                retriever_filepath: str, num_sequences: int, at_k: int, category: str):
    """Main function to load data, evaluate retrieval, and print results."""
    print("Starting the evaluation process...")

    # Load data
    print("Loading data...")
    asins_compact, genop = load_data(meta_filepath, generated_filepath)

    # Verify reviewer IDs
    print("Verifying reviewer IDs...")
    verify_reviewer_ids(genop)

    # make denser retrieval index if not already made
    if not os.path.exists(retriever_filepath):
        print("Making denser retrieval index...")
        create_dense_retrieval_index(meta_filepath, retriever_filepath)
    else:
        print("Denser retrieval index already exists...")
        dr = DenseRetriever.load(retriever_filepath)

    # Evaluate retrieval
    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences, asins_compact, at_k)

    print("\n--- Evaluation Summary ---")
    print(f"Category: {category}")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print("Metrics:", ans)
