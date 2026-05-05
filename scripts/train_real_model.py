import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_field(row, keys):
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def build_text(row):
    statement = get_field(row, ["formal_statement", "statement", "goal", "theorem", "text"])
    goal = get_field(row, ["goal", "nl_statement", "informal_prefix"])
    header = get_field(row, ["header", "src_header"])
    proof = get_field(row, ["proof", "formal_proof", "solution"])
    if not statement and not goal:
        return None
    # Supervised if proof exists. Otherwise still train real LM on prover-format prompt.
    prompt = f"### Problem\n{statement or goal}\n\n"
    if header:
        prompt += f"### Header\n{header}\n\n"
    if goal:
        prompt += f"### Goal\n{goal}\n\n"
    if proof:
        return prompt + f"### Lean proof\n{proof}\n"
    return prompt + "### Lean proof\nby\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="minif2f_raw.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out-dir", default="outputs/real_model")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()

    rows = read_jsonl(args.data)
    texts = []
    n_supervised = 0
    for r in rows:
        t = build_text(r)
        if t:
            if get_field(r, ["proof", "formal_proof", "solution"]):
                n_supervised += 1
            texts.append({"text": t})
        if len(texts) >= args.limit:
            break
    if not texts:
        raise RuntimeError("No trainable rows found with statement/goal fields in dataset.")
    print(f"train rows: {len(texts)} | supervised rows: {n_supervised}")

    ds = Dataset.from_list(texts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    use_cuda = torch.cuda.is_available()
    major, _minor = (0, 0)
    if use_cuda:
        major, _minor = torch.cuda.get_device_capability(0)
    # T4 (sm75) should use fp16; bf16 is stable on Ampere+ (sm80+).
    use_bf16 = bool(use_cuda and major >= 8)
    use_fp16 = bool(use_cuda and not use_bf16)
    print(f"dtype flags: bf16={use_bf16} fp16={use_fp16}")
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=model_dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"saved model: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
