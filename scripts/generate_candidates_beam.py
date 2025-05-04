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
def generate_beam_sequences(model, tokenizer, dataset: Dataset,
                           batch_size: int = 2, num_beams: int = 5,
                           num_return_sequences: int = 5, max_context: int = 1024,
                           max_new_tokens: int = 10,
                           field='ptext') -> List[Dict]:
    """
    Generates multiple sequences using beam search for each input text in the dataset.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        dataset: The dataset containing input prompts.
        batch_size: Number of prompts to process in parallel.
        num_beams: Number of beams for beam search.
        num_return_sequences: Number of sequences to generate per prompt.
                               Must be <= num_beams.
        max_new_tokens: Maximum number of new tokens to generate.
        field: The name of the dataset column containing the input text.

    Returns:
        A list of dictionaries, where each dictionary contains:
            - "reviewer_id": The ID of the reviewer.
            - "asin": The ASIN of the product.
            - "seen_asins": The ASINs seen by the reviewer.
            - "generated_sequences": A list of generated sequences (strings).
    """
    if model.training:
        print("Warning: Model is in training mode. Setting to evaluation mode.")
        model.eval() # Ensure model is in eval mode
        
    if num_return_sequences > num_beams:
        raise ValueError(f"num_return_sequences ({num_return_sequences}) cannot be greater than num_beams ({num_beams})")

    results: List[Dict] = []
    dataset_size = len(dataset)
    loop_total_batches = (dataset_size + batch_size - 1) // batch_size
    num_batches_processed = 0
    print(f"Processing {dataset_size} items in {loop_total_batches} batches using beam search.")
    for i in tqdm(range(0, dataset_size, batch_size), desc="Generating beam sequences", unit="batch"):
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
                           max_length=max_context - max_new_tokens
                          ).to(model.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_seq_len = input_ids.shape[1] # Length of the tokenized input prompt

        # --- Updated generation_config for Beam Search ---
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,                     # Added num_beams
            "num_return_sequences": num_return_sequences,
            "do_sample": False,                         # Set do_sample to False
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_scores": False,
            "return_dict_in_generate": False
        }
        # --- 1. Generate ---
        generated_outputs_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
            use_cache=True # Cache is usually beneficial for beam search
        )

        # --- 2. Process Results (Decode, Structure) ---
        # Slice to get only the generated token IDs
        all_generated_only_ids = generated_outputs_ids[:, input_seq_len:]

        # Delete large GPU tensors early
        del generated_outputs_ids
        del inputs
        del input_ids
        del attention_mask
        torch.cuda.synchronize() # Ensure operations involving deleted tensors are complete

        # Move generated IDs to CPU for decoding
        all_decoded_texts = tokenizer.batch_decode(all_generated_only_ids.cpu(), skip_special_tokens=True)

        del all_generated_only_ids

        # Structure results for each input text in the batch
        for j, input_text in enumerate(input_texts): # input_text is not used below, but enumerate is useful
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

def gen_with_beam(model_name: str, dataset_name: str, dataset_split: str,  max_seq_length:int,
                  batch_size: int, num_beams: int, num_return_sequences: int, max_new_tokens: int=50,
                  data_field:str="ptext", load_in_4bit:bool=True):
    run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name.split('/')[-1]}"
    project_name = f"generatebeam_{model_name.split('/')[-1]}_{dataset_name}"
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
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})")
    print(f"Run URL: {wandb.run.get_url()}")

    print("Loading dataset...")
    dataset_path = str(get_hfdata_dir() / dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples. Using field '{data_field}'.")

    model_path = str(get_llama_modelsave_dir() / model_name)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.pad_token, tokenizer.truncation_side, tokenizer.padding_side) # padding side and tunrcation side will be set to left in the generate_beam_sequences call below
    results = generate_beam_sequences(model, tokenizer, full_split,\
                           batch_size= batch_size, num_beams= num_beams,\
                           num_return_sequences = num_return_sequences, max_new_tokens = max_new_tokens,
                           max_context=max_seq_length, field=data_field)
    gen_path = str(get_gen_dir_dataset(dataset_name) / f"{model_name.split('/')[-1]}_{dataset_split}_beam{num_beams}_max_seq{max_seq_length}v2.json")
    with open(gen_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_field", type=str, default="ptext")
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    args = parser.parse_args()
    gen_with_beam(model_name = args.model_name, dataset_name = args.dataset_name, \
                  dataset_split = args.dataset_split, max_seq_length = args.max_seq_length, \
                  batch_size = args.batch_size, num_beams = args.num_beams, \
                  num_return_sequences = args.num_return_sequences, max_new_tokens = args.max_new_tokens, \
                  data_field = args.data_field, load_in_4bit = args.load_in_4bit)