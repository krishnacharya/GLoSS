
from retriv import DenseRetriever
import pandas as pd
from src.utils.project_dirs import get_gen_dir_dataset, processed_data_dir, get_bm25_indexes_dir,get_minilm_index_dir,  get_peruser_metric_dataset_modelname
from src.calcirmetrics_bm25_dataset_agno import load_data, verify_reviewer_ids, get_qrels
from ranx import Qrels, Run, evaluate
from typing import List, Dict
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

def get_rundict_from_dense_retriever(genop: List[Dict], dr: DenseRetriever, num_return_sequences: int):
    run_dict = defaultdict(dict)
    l = len(genop)
    if num_return_sequences != len(genop[0]['generated_sequences']):
        raise ValueError(f"num_return_sequences {num_return_sequences} "
                         f"does not match the number of generated sequences {len(genop[0]['generated_sequences'])}")
    
    for i in tqdm(range(l), desc="dense retrieving asins"):
        reviewer_id = genop[i]['reviewer_id']
        generated_seqs = genop[i]['generated_sequences']
        queries = [{'id':f'{reviewer_id}_{j}', 'text':generated_seqs[j]} for j in range(num_return_sequences)] # hacky way because retriv expects each query id to be unique
        res = dr.msearch(queries=queries, cutoff=1, batch_size=num_return_sequences) # all for reviewer_id, returns a dictionary top asin text for each of the num_return_sequences genrated text {'rid_0': {'B00ECUPDYC': 0.66617227}, 'rid_1': {'B000052YM0': 0.72061837}}
        for k, v in res.items():
            asin, score = next(iter(v.items()))
            run_dict[reviewer_id][asin] = score
    return run_dict

def get_rundict_from_dense_faster(genop: List[Dict], dr: DenseRetriever, num_return_sequences: int, batch_size:int=None):
    if batch_size == None:
        batch_size = num_return_sequences

    run_dict = defaultdict(dict)
    l = len(genop)
    all_queries = []
    # first get all the queries
    for i in range(l):
        reviewer_id = genop[i]['reviewer_id']
        generated_seqs = genop[i]['generated_sequences']
        queries = [{'id':f'{reviewer_id}_{j}', 'text':generated_seqs[j]} for j in range(num_return_sequences)]
        all_queries.extend(queries)
    
    #batched search
    # res = dr.msearch(queries=all_queries, cutoff=1, batch_size=batch_size) # dictionary of reviewer_id_j -> {asin: score}
    res = dr.bsearch(queries=all_queries, cutoff=1, batch_size=batch_size, show_progress=True)
    # first sort by reviewer_id
    # res = dict(sorted(res.items(), key=lambda item: item[0].split('_')[0]))
    for k, v in res.items():
        k = k.split('_')[0] # the reviewer id
        asin, score = next(iter(v.items()))
        run_dict[k][asin] = score
    return run_dict



def evaluate_retrieval(genop: List[Dict], retriever_filepath: str, num_return_sequences: int, at_k: int, batch_size:int=128, metrics:List[str]=None):
    '''
        genop: list of dicts, each dict contains 
    '''
    if metrics == None:
        metrics = ["recall@" + str(at_k), "ndcg@" + str(at_k), "mrr"]
    qrels = Qrels(get_qrels(genop))
    dr = DenseRetriever.load(index_name = retriever_filepath)
    # run_dict = get_rundict_from_dense_retriever(genop, dr, num_return_sequences)
    run_dict = get_rundict_from_dense_faster(genop, dr, num_return_sequences, batch_size=batch_size)
    rundR = Run(run_dict)
    ans = evaluate(qrels, rundR, metrics) 
    return qrels, rundR, ans


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
    dr = dr.index_file(path=meta_filepath, # has to be json lines, each seperated by new line
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
                retriever_filepath: str, num_sequences: int, at_k: int, category: str,
                max_length:int=256, model_name:str='sentence-transformers/all-MiniLM-L6-v2', use_ann:bool=False,
                use_gpu:bool=True, batch_size:int=128, show_progress:bool=True,
                peruser_savepath:str=None):
    '''
        Main function to load data, evaluate retrieval, and print results.
        meta_filepath: path to the meta corpus file
        max_length: max length of the text to be embedded
        use_ann: use approximate nearest neighbor index, set to false for <20k items
        use_gpu: use GPU for index creation/embedding gen
    '''
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
        create_dense_retrieval_index(meta_filepath, retriever_filepath,\
                                      max_length, model_name, use_ann, use_gpu, batch_size, show_progress)
    else:
        print("Dense retrieval index already exists...")
        dr = DenseRetriever.load(retriever_filepath)

    # Evaluate retrieval
    print("Evaluating retrieval performance...")
    qrels, rundR, ans = evaluate_retrieval(genop, retriever_filepath, num_sequences, at_k, batch_size=batch_size)
    
    perusermetrics = rundR.scores
    df_metrics = pd.DataFrame(perusermetrics)
    df_metrics.to_csv(peruser_savepath + ".csv", index=False)
    df_metrics.to_json(peruser_savepath + ".jsonl", orient="records")


    print("\n--- Evaluation Summary ---")
    print(f"Category: {category}")
    print(f"Generated sequences file: {generated_filepath}")
    print(f"Retriever index file: {retriever_filepath}")
    print(f"Number of return sequences: {num_sequences}")
    print("Metrics:", ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences.")
    parser.add_argument("--category", type=str, required=True, help="The category of the dataset to evaluate (e.g., 'beauty', 'toys').")
    parser.add_argument("--generated_file", type=str, required=True, help="The JSON file containing generated sequences (e.g., 'val_gen_op.json').")
    parser.add_argument("--split", type=str, required=True, help="The split to evaluate on (e.g., 'validation', 'test').")
    parser.add_argument("--short_model_name", type=str, required=True, help="The short model name (e.g., 'llama-1b').")

    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")
    parser.add_argument("--at_k", type=int, default=5, help="Number of return sequences to consider per reviewer.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for dense retrieval.")

    args = parser.parse_args()
    category = args.category.lower()
    generated_filepath = str(get_gen_dir_dataset(category) / args.generated_file)
    meta_filepath = str(processed_data_dir(f'{category}2014') / 'meta_corpus.jsonl')
    retriever_filepath = str(get_minilm_index_dir() / f'{category}2014_index')

    filename = (args.category + "_" + args.split + "_" + args.short_model_name)
    # peruser_savepath = str(get_peruser_metric_dir() / args.category / args.short_model_name / filename)
    peruser_savepath = str(get_peruser_metric_dataset_modelname(args.category, args.short_model_name) / filename)

    get_metrics(meta_filepath = meta_filepath, generated_filepath = generated_filepath, \
                retriever_filepath = retriever_filepath, num_sequences = args.num_sequences, at_k = args.at_k, category = category,
                batch_size = args.batch_size, peruser_savepath = peruser_savepath)



## RESULTS on TOYS, VALIDATION
# LLama 1B with batch size 5 for dense msearch
# Metrics: {'recall@5': 0.06604747162022703, 'ndcg@5': 0.04607103004116466, 'mrr': 0.03949088407292742} 

# LLama 1B with batch size 128 for dense msearch

# ------------------------------------------------------------------------------------------------

## RESULTS on TOYS, TEST
# Llama 1B with batch size 5 for dense msearch
# Metrics: {'recall@5': 0.06774941995359629, 'ndcg@5': 0.04478732010685207, 'mrr': 0.03727593022256595}

# LLama 1B with batch size 128 for dense bsearch
# Metrics: {'recall@5': 0.06764630059293632, 'ndcg@5': 0.043830376071491484, 'mrr': 0.03603334192661339}

# LLama 1B with batch size 512 for dense bsearch

