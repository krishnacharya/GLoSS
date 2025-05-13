
from src.utils.project_dirs import get_gen_dir_model, get_bm25_indexes_dir, processed_data_dir
from scripts.calc_irmetrics_beauty2014 import get_metrics
import argparse

def main(model_name: str, dataset: str, retriever_index: str, num_sequences: int, at_k: int):
    gen_dir = get_gen_dir_model(dataset, model_name)
    meta_filepath = str(processed_data_dir(dataset) / 'meta_corpus.json')
    retriever_filepath = str(get_bm25_indexes_dir() / retriever_index)
    for json_file in gen_dir.glob("*.json"):
        print(json_file)
        get_metrics(meta_filepath, json_file, retriever_filepath, num_sequences, at_k)
        print("--------------------------------")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance based on generated sequences.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate.")
    parser.add_argument("--retriever_index", type=str, required=True, help="Name of the retriever index to evaluate.")
    parser.add_argument("--num_sequences", type=int, default=5, help="Number of generated sequences to consider per reviewer.")
    parser.add_argument("--at_k", type=int, default=5, help="Number of results to consider for evaluation.")

    args = parser.parse_args()
    model_name = args.model_name
    dataset = args.dataset
    retriever_index = args.retriever_index
    num_sequences = args.num_sequences
    at_k = args.at_k

    main(model_name, dataset, retriever_index, num_sequences, at_k)