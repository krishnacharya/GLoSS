import torch
from datasets import Dataset
from typing import List, Dict, Tuple
from unsloth import FastLanguageModel
from src.utils.project_dirs import get_hfdata_dir, project_root, get_llama_modelsave_dir, get_gen_dir_dataset
from datasets import load_from_disk
import wandb
import os
import time
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse

@torch.no_grad()
def generate_sampled_sequences(model, tokenizer, dataset: Dataset,
                       batch_size: int = 2, num_return_sequences: int = 3,
                       max_new_tokens: int = 10, temperature: float = 0.5,
                       field='ptext') -> List[Dict]:
    """
    Generates multiple sequences for each input text in the dataset.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        dataset: The dataset containing input prompts.
        batch_size: Number of prompts to process in parallel.
        num_return_sequences: Number of sequences to generate per prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        field: The name of the dataset column containing the input text.

    Returns:
        A list of dictionaries, where each dictionary contains:
            - "input_text": The original input prompt.
            - "reviewer_id": The ID of the reviewer.
            - "asin": The ASIN of the product.
            - "seen_asins": The ASINs seen by the reviewer in val it's the n-2 items, in test it's the n-1 items.
            - "generated_sequences": A list of generated sequences (strings).
    """
    if model.training:
        print("Warning: Model is in training mode. Setting to evaluation mode.")
        model.eval() # Ensure model is in eval mode

    results: List[Dict] = []
    dataset_size = len(dataset)
    loop_total_batches = (dataset_size + batch_size - 1) // batch_size
    num_batches_processed = 0
    print(f"Processing {dataset_size} items in {loop_total_batches} batches.")
    for i in tqdm(range(0, dataset_size, batch_size), desc="Generating sequences", unit="batch"):
        batch_start_idx = i
        batch_end_idx = min(i + batch_size, dataset_size)
        current_batch_indices = range(batch_start_idx, batch_end_idx)

        if not current_batch_indices: continue
        input_texts = dataset[current_batch_indices][field] # list of input texts

        current_batch_size = len(input_texts)
        if current_batch_size == 0: continue
        num_batches_processed += 1

        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left" # remove earlier tokens if the input is too long
        inputs = tokenizer(input_texts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=model.max_seq_length - max_new_tokens
                          ).to(model.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_seq_len = input_ids.shape[1] # Length of the tokenized input prompt

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_scores": False,
            "return_dict_in_generate": False
        }
        # --- 1. Generate ---, slow for large models, HF KV cache is also slow
        generated_outputs_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
            use_cache=True
        )

        # --- 2. Process Results (Decode, Structure) ---
        # Slice to get only the generated token IDs (shape: [batch_size * num_return_sequences, generated_seq_len])
        all_generated_only_ids = generated_outputs_ids[:, input_seq_len:]

        # Delete large GPU tensors early
        del generated_outputs_ids
        del inputs
        del input_ids
        del attention_mask

        # Move generated IDs to CPU for decoding
        all_decoded_texts = tokenizer.batch_decode(all_generated_only_ids.cpu(), skip_special_tokens=True)

        del all_generated_only_ids

        # Structure results for each input text in the batch
        for j, input_text in enumerate(input_texts):
            original_dataset_index = batch_start_idx + j
            start_decode_idx = j * num_return_sequences
            end_decode_idx = (j + 1) * num_return_sequences
            generated_texts: List[str] = all_decoded_texts[start_decode_idx:end_decode_idx]

            # Use the correct index to fetch metadata
            results.append({
                "reviewer_id": dataset[original_dataset_index]['reviewer_id'],
                "asin": dataset[original_dataset_index]['asin'],
                "seen_asins": dataset[original_dataset_index]['seen_asins'],
                "generated_sequences": generated_texts,
            })
        # --- Explicitly clear cache and collect garbage ---
        torch.cuda.empty_cache()
        gc.collect()
    return results

def generate_with_sampling(model_name: str, dataset_name: str, dataset_split: str, max_seq_length: int,
                           batch_size: int, num_return_sequences: int, max_new_tokens: int = 50,
                           data_field: str = "ptext", temperature: float = 1.0, load_in_4bit: bool = True):
    run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name.split('/')[-1]}_temp{str(temperature).replace('.', 'p')}"
    project_name = f"generatesample_{model_name.split('/')[-1]}_{dataset_name}_temp{str(temperature).replace('.', 'p')}"
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "data_field": data_field,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})")
    print(f"Run URL: {wandb.run.get_url()}")

    print("Loading dataset...")
    dataset_path = str(get_hfdata_dir() / dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples. Using field '{data_field}'.")

    try:
        model_path = str(project_root() / model_name)  # if model is local
        if not os.path.exists(model_path):
            raise FileExistsError(f"Model path {model_path} does not exist locally")
    except FileExistsError:
        model_path = model_name # use the one from HF
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_path,
                                                       max_seq_length=max_seq_length,
                                                       dtype=None,
                                                       load_in_4bit=load_in_4bit)
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    results = generate_sampled_sequences(model, tokenizer, dataset=full_split,
                                       batch_size=batch_size, num_return_sequences=num_return_sequences,
                                       max_new_tokens=max_new_tokens, temperature=temperature,
                                       field=data_field)
    gen_path = str(get_gen_dir_dataset(dataset_name) / f"{model_name.split('/')[-1]}_{dataset_split}_temp{str(temperature).replace('.', 'p')}_max_seq{max_seq_length}.json")
    with open(gen_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path or HuggingFace name of the model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset directory in HF_DATA_DIR")
    parser.add_argument("--dataset_split", type=str, default="validation", help="Split of the dataset to use (e.g., train, validation, test)")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for the model")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to generate per input")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--data_field", type=str, default="ptext", help="Name of the field containing the input text in the dataset")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Whether to load the model in 4-bit precision")
    args = parser.parse_args()
    generate_with_sampling(model_name = args.model_name, dataset_name = args.dataset_name,
                  dataset_split = args.dataset_split, max_seq_length = args.max_seq_length,
                  batch_size = args.batch_size, num_return_sequences = args.num_return_sequences,
                  max_new_tokens = args.max_new_tokens, data_field = args.data_field,
                  temperature = args.temperature, load_in_4bit = args.load_in_4bit)