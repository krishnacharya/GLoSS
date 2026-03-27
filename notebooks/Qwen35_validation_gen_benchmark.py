#!/usr/bin/env python3
"""
Minimal check: Qwen3.5-4B text-only chat + **left-padded batch** (two prompts).

Runs two user turns with different lengths so you can confirm shapes / attention_mask.
Pass your own strings with ``--prompts`` (two or more).

Run:
  python notebooks/Qwen35_validation_gen_benchmark.py
"""

from __future__ import annotations

import argparse

import torch
from unsloth import FastLanguageModel

# Short vs long so padded width and mask differ meaningfully between rows.
DEFAULT_PROMPTS = [
    "What is 2+2? One short sentence.",
    "List three European capital cities and one fact about each; keep each fact under 15 words.",
]


def _user_message(text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
    }


def encode_chat_prompt(tokenizer, user_text: str) -> torch.Tensor:
    """Single sequence: chat template → 1D token ids (no padding)."""
    out = tokenizer.apply_chat_template(
        [_user_message(user_text)],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Unwrap BatchEncoding / dict; HF often returns [1, seq] not [seq].
    if isinstance(out, dict):
        t = out["input_ids"]
    elif hasattr(out, "input_ids"):
        t = out.input_ids
    else:
        t = out
    t = t.long()
    if t.dim() == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t.reshape(-1)


def left_pad_batch(
    rows: list[torch.Tensor], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    t_max = max(int(r.shape[0]) for r in rows)
    batch = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), t_max), dtype=torch.long, device=device)
    batch = batch.to(device)
    for i, r in enumerate(rows):
        r = r.reshape(-1).to(device)
        L = int(r.shape[0])
        batch[i, -L:] = r
        attn[i, -L:] = 1
    return batch, attn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="unsloth/Qwen3.5-4B")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="User texts (default: two built-ins: short + long).",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    prompts = args.prompts if args.prompts else list(DEFAULT_PROMPTS)

    kw = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    if not args.load_in_4bit:
        kw["load_in_16bit"] = True
        kw["full_finetuning"] = False

    model, tokenizer = FastLanguageModel.from_pretrained(**kw)
    FastLanguageModel.for_inference(model)

    device = model.device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    per_row = [encode_chat_prompt(tokenizer, t) for t in prompts]
    print("Per-example prompt lengths (tokens, before pad):")
    for i, row in enumerate(per_row):
        print(f"  [{i}] len={row.shape[0]} | {prompts[i][:80]!s}...")

    input_ids, attention_mask = left_pad_batch(per_row, int(pad_id), device)
    print(
        f"\nBatched input_ids: {tuple(input_ids.shape)} | "
        f"attention_mask sum per row: {attention_mask.sum(dim=1).tolist()}"
    )
    print("(Left padding: shorter prompt has more leading pad tokens; sums should match raw lengths.)\n")

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    plen = input_ids.shape[1]
    for i in range(len(prompts)):
        text = tokenizer.decode(out[i, plen:], skip_special_tokens=True)
        print(f"--- Generated [{i}] ---\n{text.strip()}\n")


if __name__ == "__main__":
    main()
