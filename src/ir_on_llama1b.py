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
from src.utils.project_dirs import get_hfdata_dir, get_reviews_raw2018_dir
# from src.utils.generate_utils import generate_sequences
# from unsloth import FastLanguageModel
from collections import defaultdict

def get_qrels(genop):
    '''
        Returns: dict[reviewer_id] = {asin1: 1, asin2: 1, ...}
    '''
    qrels_dict = defaultdict(dict)
    for row in genop:
        reviewer = row['reviewer_id']
        asin = row['asin']
        qrels_dict[reviewer][asin] = 1
    return dict(qrels_dict)

def get_rundict(genop:List[Dict], retriever, num_return_sequences: int, asins_compact: pd.DataFrame):
    run_dict = {}
    l = len(genop)

    if num_return_sequences != len(valgen[0]['generated_sequences']):
        raise ValueError(f"num_return_sequences {num_return_sequences} \
                         does not match the number of generated sequences {len(valgen[0]['generated_sequences'])}")

    # Flatten all generated queries
    queries_flat = []
    for row in genop:
        for seq in row['generated_sequences']:
            queries_flat.append(seq)
    
    # Tokenize and retrieve
    stemmer = Stemmer.Stemmer("english")
    # query_tokens = bm25s.tokenize(queries_flat, stemmer=stemmer)
    query_tokens = bm25s.tokenize(queries_flat, stopwords="en")
    res, scores = retriever.retrieve(query_tokens, k=1)
    print(res.shape, scores.shape)
    
    # Reshape to group by reviewer
    res_temp = res.reshape((l, num_return_sequences))
    scores_temp =  scores.reshape((l, num_return_sequences))

    res = res_temp
    scores = scores_temp

    # Helper: return unique asins sorted by score
    def get_unique_sorted_asins(asins, scores):
        seen = set()
        asin_score_pairs = []
        for asin, score in sorted(zip(asins, scores), key=lambda x: -x[1]):
            if asin not in seen:
                asin_score_pairs.append((asin, score))
                seen.add(asin)
        return dict(asin_score_pairs)

    for i in range(l):
        reviewer_id = genop[i]['reviewer_id']
        asin_indices = res[i]
        asin_scores = scores[i]
        asins = asins_compact.iloc[asin_indices]['asin'].tolist()
        run_dict[reviewer_id] = get_unique_sorted_asins(asins, asin_scores)

    return run_dict

if __name__=="__main__":
    filepath_scimeta = str(get_reviews_raw2018_dir() / "scimeta_corpus.json")
    scimeta = pd.read_json(filepath_scimeta, orient='records', lines=True)
    scimeta.columns = ['asin', 'Title', 'Brand', 'Category', 'Price']
    asins_compact = scimeta[['asin']].copy()
    asins_compact['nlang'] = (
        "Title: " + scimeta['Title'] + ". " +
        "Brand: " + scimeta['Brand'] + ". " +
        "Category: " + scimeta['Category'] + ". " +
        "Price: " + scimeta['Price']
    )
    asin_dict = asins_compact.set_index('asin')['nlang'].to_dict()

    # filepath_valgen = str(get_reviews_raw2018_dir() / "llama1B_val_no-ft.json")
    # filepath_valgen = str(get_reviews_raw2018_dir() / "llama1B_val_noft_beam5.json")
    # filepath_valgen = str(get_reviews_raw2018_dir() / "llama3B_val_no-ft.json")
    # filepath_valgen = str(get_reviews_raw2018_dir() / "llama3B_val_noft_beam5.json")
    # filepath_valgen = str(get_reviews_raw2018_dir() / "llama8B_ft_beam5.json")
    filepath_valgen = str(get_reviews_raw2018_dir() / "llama8B_val_ft_temp05.json")

    with open(filepath_valgen, "r") as f:
        valgen = json.load(f)
    print(type(valgen), len(valgen[0]))

    reviewers = [row['reviewer_id'] for row in valgen]
    print(len(reviewers))                 # Should be 512
    print(len(set(reviewers)))           # If this is 4, they’re not unique

    qrels_val = Qrels(get_qrels(valgen))

    filepath_retriever = str(get_reviews_raw2018_dir() / "amznsci_2018_index2")
    retriever = bm25s.BM25.load(filepath_retriever, load_corpus=False)
    run_dict = get_rundict(valgen, retriever, num_return_sequences=5, asins_compact=asins_compact)
    
    rundR = Run(run_dict)

    ans = evaluate(qrels_val, rundR, ["recall@1", "recall@5", "ndcg@5", "mrr@5"])
    print(ans)

# ----- no finetuning just prompted -----
# with llama1B with 5 gen items sampled temp 0.5
# {'recall@5': np.float64(0.009765625), 'ndcg@5': np.float64(0.008323944349888507), 'mrr@5': np.float64(0.0078125)}

# with llama1B with 5 beam search
# {'recall@5': np.float64(0.005859375), 'ndcg@5': np.float64(0.004026574827431349), 'mrr@5': np.float64(0.00341796875)}

# with llama3B with 5 beam search
# {'recall@5': np.float64(0.0078125), 'ndcg@5': np.float64(0.005258859502375602), 'mrr@5': np.float64(0.00439453125)}

# with llama3B  with 5 gen items sampled temp 0.5
# {'recall@5': np.float64(0.0078125), 'ndcg@5': np.float64(0.006115097174944253), 'mrr@5': np.float64(0.005533854166666666)}


# ----- FINETUNED llama8B promising ----- takes around 30min, need faster generation use llama1B finetuned
# with llama8B finetuned, gen with parallel sampling temp 0.5
# {'recall@1': np.float64(0.033203125), 'recall@5': np.float64(0.052734375), 'ndcg@5': np.float64(0.044011297743769676), 'mrr@5': np.float64(0.04108072916666666)}

# with llama8B finetuned, gen with 5 beam search
# {'recall@1': np.float64(0.029296875), 'recall@5': np.float64(0.083984375), 'ndcg@5': np.float64(0.058760452051613565), 'mrr@5': np.float64(0.05032552083333333)}

