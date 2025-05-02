import torch
from datasets import Dataset
from typing import List, Dict
from unsloth import FastLanguageModel
from src.utils.project_dirs import get_hfdata_dir
from datasets import load_from_disk

def generate_and_rank_sequences(model, tokenizer, dataset: Dataset, batch_size: int = 2, \
                                num_return_sequences: int = 3, max_new_tokens: int = 10, temperature: float = 1.0) -> List[Dict]:
    """
    Generates sequences from a language model in batches, calculates log likelihoods, ranks them, and removes duplicates.

    Args:
        model_name: The name of the Hugging Face model.
        dataset: A Hugging Face dataset.
        batch_size: The batch size for input datapoints.
        num_return_sequences: The number of sequences to generate per input.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature for generation.

    Returns:
        A list of dictionaries, where each dictionary contains the input text, generated sequences with their log likelihoods, and the ranked unique sequences.
    """
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    # check if model is in eval mode, give warning if not
    results: List[Dict] = []
    dataset_size = len(dataset)
    for i in range(0, dataset_size, batch_size):
        input_texts = dataset[i:i + batch_size]['text']

        # hacky but important to have these correct
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left" # if required to truncate remove the start of the sequence
        tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device) # shape of inputs.input_ids is batchsize x maxseqin that batch

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences":num_return_sequences,
            "temperature":temperature,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
        }

        generated_outputs = model.generate(**inputs, **generation_config) # has shape (batch_size * num_return_sequences, inputs.input_ids.shape[1] + max_new_tokens)
        generated_texts_batched = [tokenizer.batch_decode(generated_outputs[j * num_return_sequences:(j + 1) * num_return_sequences], \
                                                          skip_special_tokens=True) for j in range(len(input_texts))] # list of size batch size[[], []], each inner list of size num_return_sequences

        for j, input_text in enumerate(input_texts):
            generated_texts = [text[len(input_text):] for text in generated_texts_batched[j]] # goes through the each of num_return_sequences and removes the input text

            log_likelihoods = []
            for generated_text in generated_texts:
                full_text = input_text + generated_text
                full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(full_ids, labels=full_ids)
                    log_likelihood = outputs.loss.item() * (full_ids.shape[1] -1) # adjust for token count
                log_likelihoods.append(log_likelihood)

            generated_with_loglikelihood = list(zip(generated_texts, log_likelihoods))
            ranked_sequences = sorted(generated_with_loglikelihood, key=lambda x: x[1])

            unique_ranked_sequences = []
            seen_sequences = set()
            for seq, ll in ranked_sequences:
                if seq not in seen_sequences:
                    unique_ranked_sequences.append((seq, ll))
                    seen_sequences.add(seq)

            results.append({
                "input_text": input_text,
                "generated_sequences": generated_with_loglikelihood,
                "ranked_unique_sequences": unique_ranked_sequences,
            })

    return results

if __name__=="__main__":
    max_seq_length = 8192 # largest tokenized sequence i checked was around 5k, so this is works
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print(tokenizer.padding_side, tokenizer.truncation_side, \
        tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, \
        tokenizer.max_len_single_sentence
        )
    dataset = load_from_disk(get_hfdata_dir() / "Amzn_scientific_2018")
    
    results = generate_and_rank_sequences(model, tokenizer, dataset=dataset['validation'], batch_size = 2, num_return_sequences = 3, max_new_tokens = 10, temperature = 0.5)
    print(results)