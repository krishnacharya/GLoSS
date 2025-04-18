import torch
from datasets import Dataset
from typing import List, Dict, Tuple
from unsloth import FastLanguageModel
from src.utils.project_dirs import get_hfdata_dir, project_root
from datasets import load_from_disk
import wandb
import os
import time
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


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
        demo: If True, run on a small subset of the dataset.
        num_demo_prompt: Number of prompts to use in demo mode.
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

    # Ensure num_return_sequences is valid for beam search
    if num_return_sequences > num_beams:
        raise ValueError(f"num_return_sequences ({num_return_sequences}) cannot be greater than num_beams ({num_beams})")

    results: List[Dict] = []
    dataset_size = len(dataset)
    loop_total_batches = (dataset_size + batch_size - 1) // batch_size
    num_batches_processed = 0
    print(f"Processing {dataset_size} items in {loop_total_batches} batches using beam search.")
    # Changed tqdm description
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

def gen_with_llama1b_beam():
    model_name= "unsloth/Llama-3.2-1B"
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "validation"
    data_field = 'ptext'
    max_seq_length = 1024

    batch_size = 8
    num_beams = 5
    num_return_sequences = 5
    max_new_tokens = 50

    run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name.split('/')[-1]}"

    wandb.init(
        project="llama3.2-beamsearch1B", # Or your desired project name
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
    dataset_path = os.path.join(get_hfdata_dir(),dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples. Using field '{data_field}'.")


    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = generate_beam_sequences(model, tokenizer, full_split,\
                           batch_size= batch_size, num_beams= num_beams,\
                           num_return_sequences = num_return_sequences, max_new_tokens = max_new_tokens, 
                           max_context=1024, field=data_field)
    with open("llama1B_val_noft_beam5.json", "w") as f:
        json.dump(results, f, indent=2)

def gen_with_llama3b_beam():
    model_name= "unsloth/Llama-3.2-3B"
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "validation"
    data_field = 'ptext'
    max_seq_length = 1024

    batch_size = 8
    num_beams = 5
    num_return_sequences = 5
    max_new_tokens = 50

    run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name.split('/')[-1]}"

    wandb.init(
        project="llama3.2-beamsearch3B", # Or your desired project name
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
    dataset_path = os.path.join(get_hfdata_dir(),dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples. Using field '{data_field}'.")


    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = generate_beam_sequences(model, tokenizer, full_split,\
                           batch_size= batch_size, num_beams= num_beams,\
                           num_return_sequences = num_return_sequences, max_new_tokens = max_new_tokens, 
                           max_context=1024, field=data_field)
    with open("llama3B_val_noft_beam5.json", "w") as f:
        json.dump(results, f, indent=2)


def gen_with_ftllama8b_sampledseq():
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "validation"
    data_field = 'ptext'
    max_seq_length = 1024
    batch_size = 2
    temperature = 0.5
    num_return_sequences = 5
    max_new_tokens = 50

    dataset_path = os.path.join(get_hfdata_dir(), dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples.")

    model_name = "llama8B-ft"
    run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name}"
    wandb.init(
        project="llama8B-ft-temp05-sampled", # Or your desired project name
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

    model_name = str(project_root() / 'output_vramcheck/checkpoint-685')  # TODO hacky rn
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_name,\
            max_seq_length = max_seq_length,\
            dtype = None,\
            load_in_4bit = True)
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    results = generate_sampled_sequences(model, tokenizer, dataset = full_split,
                       batch_size  = batch_size, num_return_sequences = num_return_sequences,
                       max_new_tokens=max_new_tokens, temperature = temperature,
                       field=data_field)
    with open("llama8B_val_ft_temp05.json", "w") as f:
        json.dump(results, f, indent=2)

def gen_with_ftllama8b_beamseq():
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "validation"
    data_field = 'ptext'
    max_seq_length = 1024
    batch_size = 4
    num_return_sequences = 5
    num_beams = 5
    max_new_tokens = 50

    dataset_path = os.path.join(get_hfdata_dir(), dataset_name)
    full_dataset = load_from_disk(dataset_path)
    full_split = full_dataset[dataset_split]
    print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples.")

    model_name_or_path = str(project_root() / 'output_vramcheck/checkpoint-685')  # TODO hacky rn
    run_name = f"gen_beam{num_beams}_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_ftllama8b"
    wandb.init(
        project="llama8B-ft-beams5", 
        name=run_name,
        config={
            "model_path": model_name_or_path,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "data_field": data_field,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "num_return_sequences": num_return_sequences,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})")
    print(f"Run URL: {wandb.run.get_url()}")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    results = generate_beam_sequences(model, tokenizer, full_split,
                           batch_size=batch_size, num_beams=num_beams,
                           num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens,
                           max_context=max_seq_length, field=data_field)

    with open(f"llama8B_ft_beam{num_beams}.json", "w") as f:
        json.dump(results, f, indent=2)

# def gen_with_ftllama8b_beamseq():
#     dataset_name = "Amzn_scientific_2018"
#     dataset_split = "validation"
#     data_field = 'ptext'
#     max_seq_length = 1024
#     batch_size = 2
#     num_return_sequences = 5
#     num_beams = 5
#     max_new_tokens = 50

#     dataset_path = os.path.join(get_hfdata_dir(), dataset_name)
#     full_dataset = load_from_disk(dataset_path)
#     full_split = full_dataset[dataset_split]
#     print(f"Loaded dataset '{dataset_name}', split '{dataset_split}' with {len(full_split)} examples.")

#     model_name = "llama8B-ft"
#     run_name = f"gen_bs{batch_size}_numret{num_return_sequences}_maxlen{max_new_tokens}_{model_name}"
#     wandb.init(
#         project="llama8B-ft-beam5", # Or your desired project name
#         name=run_name,
#         config={
#             "model_name": model_name,
#             "dataset_name": dataset_name,
#             "dataset_split": dataset_split,
#             "data_field": data_field,
#             "max_seq_length": max_seq_length,
#             "batch_size": batch_size,
#             "num_return_sequences": num_return_sequences,
#             "max_new_tokens": max_new_tokens,
#         }
#     )
#     print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})")
#     print(f"Run URL: {wandb.run.get_url()}")

#     model_name = str(project_root() / 'output_vramcheck/checkpoint-685')  # TODO hacky rn
#     model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_name,\
#             max_seq_length = max_seq_length,\
#             dtype = None,\
#             load_in_4bit = True)
#     FastLanguageModel.for_inference(model) # Enable native 2x faster inference

#     results =  generate_beam_sequences(model, tokenizer, full_split,
#                            batch_size=batch_size, num_beams = num_beams,
#                            num_return_sequences = num_return_sequences, max_context=1024,
#                            max_new_tokens=max_new_tokens,
#                            field='ptext') -> List[Dict]:
#     with open("llama8B_ft_beam5.json", "w") as f:
#         json.dump(results, f, indent=2)




if __name__=="__main__":
    # gen_with_llama1b_beam()
    # gen_with_llama3b_beam()
    # gen_with_ftllama8b_sampledseq()
    gen_with_ftllama8b_beamseq()
