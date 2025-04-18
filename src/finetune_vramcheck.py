import os
import torch
from datasets import Dataset
from typing import List, Dict, Tuple
from unsloth import FastLanguageModel
from src.utils.project_dirs import get_hfdata_dir
from datasets import load_from_disk
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def add_eos_to_text(example, tokenizer):
    return {"text": example["text"] + tokenizer.eos_token}

def finetune_llama1b_quantized():
    model_name = "unsloth/Llama-3.2-1B"
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "train"
    max_seq_length = 4096
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # Finetuning parameters
    lora_r = 16
    lora_alpha = 16
    learning_rate = 2e-4
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps # TODO change for multiple devices

    # --- Wandb Setup ---
    run_name = f"bs{effective_batch_size}_r{lora_r}_alpha{lora_alpha}_lr{learning_rate}_{model_name.split('/')[-1]}"
    wandb.init(project="llama3.2-1B-ft",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})") # Use wandb.run.project
    print(f"Run URL: {wandb.run.get_url()}")

    # --- Model and Tokenizer Loading ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if tokenizer.pad_token is None:
        raise ValueError("pad_token is None. Please set pad_token in the tokenizer.")

    print(f"Tokenizer Info: padding_side={tokenizer.padding_side}, "
        f"truncation_side={tokenizer.truncation_side}, "
        f"pad_token='{tokenizer.pad_token}', "
        f"eos_token='{tokenizer.eos_token}', "
        f"bos_token='{tokenizer.bos_token}'")
    
    # Loading dataset
    full_dataset = load_from_disk(os.path.join(get_hfdata_dir(), dataset_name))
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']

    train_dataset = train_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, #added eval dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # using False as it's buggy and doesnt count correct number of examples
    args = TrainingArguments(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "op_ft_llama1B",
        report_to = "wandb", # Use this for WandB etc
        evaluation_strategy = "steps", #added evaluation strategy
        eval_steps = 100, #added eval steps
        ),
    )
    trainer_stats = trainer.train()
    model.save_pretrained("llama-1b_ft")
    tokenizer.save_pretrained("llama-1b_ft")

def finetune_llama3b_quantized():
    model_name = "unsloth/Llama-3.2-3B"
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "train"
    max_seq_length = 4096
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # Finetuning parameters
    lora_r = 16
    lora_alpha = 16
    learning_rate = 2e-4
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps # TODO change for multiple devices

    # --- Wandb Setup ---
    run_name = f"bs{effective_batch_size}_r{lora_r}_alpha{lora_alpha}_lr{learning_rate}_{model_name.split('/')[-1]}"
    wandb.init(project="llama3.2-3B-ft",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})") # Use wandb.run.project
    print(f"Run URL: {wandb.run.get_url()}")

    # --- Model and Tokenizer Loading ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if tokenizer.pad_token is None:
        raise ValueError("pad_token is None. Please set pad_token in the tokenizer.")

    print(f"Tokenizer Info: padding_side={tokenizer.padding_side}, "
        f"truncation_side={tokenizer.truncation_side}, "
        f"pad_token='{tokenizer.pad_token}', "
        f"eos_token='{tokenizer.eos_token}', "
        f"bos_token='{tokenizer.bos_token}'")
    
    # Loading dataset
    full_dataset = load_from_disk(os.path.join(get_hfdata_dir(), dataset_name))
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']

    train_dataset = train_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(add_eos_to_text, fn_kwargs={"tokenizer": tokenizer})

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, #added eval dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # using False as it's buggy and doesnt count correct number of examples
    args = TrainingArguments(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "op_ft_llama3B",
        report_to = "wandb", # Use this for WandB etc
        evaluation_strategy = "steps", #added evaluation strategy
        eval_steps = 100, #added eval steps
        ),
    )
    trainer_stats = trainer.train()
    model.save_pretrained("llama-3b_ft")
    tokenizer.save_pretrained("llama-3b_ft")



def finetune_llama8b_quantized(): # TODO change EOS addition
    model_name = "unsloth/Meta-Llama-3.1-8B"
    dataset_name = "Amzn_scientific_2018"
    dataset_split = "train"
    max_seq_length = 4096
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # Finetuning parameters
    batch_size = 2  
    lora_r = 16
    lora_alpha = 16
    max_steps = 20
    learning_rate = 2e-4
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps

    # --- Wandb Setup ---
    run_name = f"ebs{effective_batch_size}_r{lora_r}_alpha{lora_alpha}_lr{learning_rate}_{model_name.split('/')[-1]}"
    wandb.init(project="llama3.2-8B-ft",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps, # Adding gradient accumulation steps to wandb config
            "effective_batch_size": effective_batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name} (Project: {wandb.run.project})") # Use wandb.run.project
    print(f"Run URL: {wandb.run.get_url()}")

    # --- Model and Tokenizer Loading ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    if tokenizer.pad_token is None:
        raise ValueError("pad_token is None. Please set pad_token in the tokenizer.")

    print(f"Tokenizer Info: padding_side={tokenizer.padding_side}, "
        f"truncation_side={tokenizer.truncation_side}, "
        f"pad_token='{tokenizer.pad_token}', "
        f"eos_token='{tokenizer.eos_token}', "
        f"bos_token='{tokenizer.bos_token}'")

    # Loading dataset
    full_dataset = load_from_disk(os.path.join(get_hfdata_dir(), dataset_name))
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']
    train_dataset = train_dataset.map(add_eos_to_text) # TODO fix with new formatting
    eval_dataset = eval_dataset.map(add_eos_to_text)

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, #added eval dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # using False as it's buggy and doesnt count correct number of examples, but with True upto 5x faster
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 10,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "output_vramcheck",
        report_to = "wandb", # Use this for WandB etc
        evaluation_strategy = "steps", #added evaluation strategy
        eval_steps = 100, #added eval steps
        ),
    )
    trainer_stats = trainer.train()

if __name__=="__main__":
    # finetune_llama8b_quantized()
    # finetune_llama1b_quantized()
    finetune_llama3b_quantized()