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
from unsloth import unsloth_train
from src.utils.project_dirs import get_hfdata_dir, get_llama_modelsave_dir

def add_eos_to_text(example, tokenizer):
    return {"text": example["text"] + tokenizer.eos_token}

def finetune_llama1b_quantized(model_name: str, dataset_name: str, dataset_split: str,
                                max_seq_length: int=4096, dtype: str=None, load_in_4bit: bool=True,
                                lora_r: int=16, lora_alpha: int=16, learning_rate: float=2e-4,
                                per_device_train_batch_size: int=4, gradient_accumulation_steps: int=4,
                                random_state: int=3407, num_train_epochs: int=1):
    
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    # --- Wandb Setup ---
    project_name = f"{model_name.split('/')[-1]}-ft{dataset_name}"
    run_name = f"bs{effective_batch_size}_r{lora_r}_alpha{lora_alpha}_lr{learning_rate}_noeval"
    wandb.init(project=project_name,
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
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if tokenizer.pad_token is None:
        raise ValueError("pad_token is None. Please set pad_token in the tokenizer.")
    
    tokenizer.truncation_side = "left" # removes tokens from the left
    tokenizer.padding_side = "right" # pad tokens on the right, https://github.com/huggingface/transformers/issues/34842#issuecomment-2490994584

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

    # model output directory
    model_output_dir = get_llama_modelsave_dir() / f"{model_name.split('/')[-1]}-ft{dataset_name.split('_')[-1]}"  # this is a pathlib object

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, #added eval dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # using False as it's buggy and doesnt count correct number of examples
    args = TrainingArguments(output_dir = str(model_output_dir),
        run_name = run_name,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = random_state,
        report_to = "wandb"
        # evaluation_strategy = "steps",
        # eval_steps = 100,
        ),
    )
    trainer_stats = unsloth_train(trainer)
    model.save_pretrained(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

if __name__ == "__main__":
    finetune_llama1b_quantized(model_name="unsloth/Llama-3.2-1B", dataset_name="beauty2014", dataset_split="train",
                                max_seq_length=4096, dtype=None, load_in_4bit=True,
                                lora_r=16, lora_alpha=16, learning_rate=2e-4,
                                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                                random_state=3407, num_train_epochs=1)

