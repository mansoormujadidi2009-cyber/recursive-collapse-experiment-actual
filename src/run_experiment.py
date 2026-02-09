import os, json, math, gc, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import statsmodels.formula.api as smf


def clean_text_list(texts, min_chars=200):
    out = []
    for t in texts:
        t = (t or "").replace("\n", " ").strip()
        if len(t) >= min_chars:
            out.append(t)
    return out

def make_prompts(texts, n_prompts=200, prompt_chars=160):
    prompts = []
    for t in texts:
        if len(prompts) >= n_prompts:
            break
        prompts.append(t[:prompt_chars])
    return prompts

def make_mixed_texts(human_texts, synth_texts, synth_ratio, total_docs, seed):
    rng = np.random.default_rng(seed)
    n_synth = int(total_docs * synth_ratio)
    n_human = total_docs - n_synth

    human_idx = rng.choice(len(human_texts), size=n_human, replace=False)
    synth_idx = rng.choice(len(synth_texts), size=n_synth, replace=False) if n_synth > 0 else []

    mixed = [human_texts[i] for i in human_idx] + ([synth_texts[i] for i in synth_idx] if n_synth > 0 else [])
    rng.shuffle(mixed)
    return mixed, n_human, n_synth

def texts_to_lm_blocks(tokenizer, texts, block_size):
    enc = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    n_blocks = len(input_ids) // block_size
    input_ids = input_ids[: n_blocks * block_size]
    blocks = input_ids.view(n_blocks, block_size)
    ds = Dataset.from_dict({"input_ids": [b.tolist() for b in blocks]})
    return ds

def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

def perplexity_from_eval_loss(eval_loss):
    try:
        return float(math.exp(float(eval_loss)))
    except OverflowError:
        return float("inf")

def distinct_n(texts, n=1):
    total = 0
    uniq = set()
    for t in texts:
        toks = (t or "").split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            total += 1
            uniq.add(tuple(toks[i:i+n]))
    return (len(uniq) / total) if total > 0 else 0.0

def next_segment_selection_accuracy_causal(model, tokenizer, texts, K=4, prompt_chars=200, cont_chars=200, n_items=120, seed=0):
    rng = np.random.default_rng(seed)
    pool = [t for t in texts if len(t) >= prompt_chars + cont_chars + 50]
    if len(pool) < n_items:
        n_items = max(10, len(pool))

    chosen = rng.choice(len(pool), size=n_items, replace=False)
    correct = 0
    model.eval()

    with torch.no_grad():
        for idx in chosen:
            t = pool[idx]
            prompt = t[:prompt_chars]
            true_cont = t[prompt_chars:prompt_chars + cont_chars]

            negs = []
            while len(negs) < (K - 1):
                j = int(rng.integers(0, len(pool)))
                if j == idx:
                    continue
                tj = pool[j]
                negs.append(tj[prompt_chars:prompt_chars + cont_chars])

            cands = [true_cont] + negs
            rng.shuffle(cands)
            true_pos = cands.index(true_cont)

            scores = []
            for c in cands:
                full = prompt + c
                enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                input_ids = enc["input_ids"]
                out = model(input_ids=input_ids)
                logits = out.logits[:, :-1, :]
                labels = input_ids[:, 1:]

                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
                p_len = len(prompt_ids)
                start = max(0, p_len - 1)

                cont_logits = logits[:, start:, :]
                cont_labels = labels[:, start:]

                logp = F.log_softmax(cont_logits, dim=-1)
                token_logp = torch.gather(logp, 2, cont_labels.unsqueeze(-1)).squeeze(-1)
                avg_logp = token_logp.mean().item()
                scores.append(avg_logp)

            pred = int(np.argmax(scores))
            if pred == true_pos:
                correct += 1

    return correct / n_items if n_items > 0 else 0.0


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids):
        x = self.embed(ids)
        out, _ = self.lstm(x)
        return self.fc(out)


def make_torch_batches_from_blocks(block_ds, batch_size, device):
    ids = torch.tensor(block_ds["input_ids"], dtype=torch.long)
    x = ids[:, :-1]
    y = ids[:, 1:]
    perm = torch.randperm(x.size(0))
    x, y = x[perm], y[perm]
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i+batch_size].to(device)
        yb = y[i:i+batch_size].to(device)
        yield xb, yb

def eval_lstm_loss(model, eval_blocks, batch_size, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in make_torch_batches_from_blocks(eval_blocks, batch_size, device):
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="mean")
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")

def next_segment_selection_accuracy_lstm(model, tokenizer, texts, K=4, prompt_chars=200, cont_chars=200, n_items=120, seed=0):
    rng = np.random.default_rng(seed)
    pool = [t for t in texts if len(t) >= prompt_chars + cont_chars + 50]
    if len(pool) < n_items:
        n_items = max(10, len(pool))

    chosen = rng.choice(len(pool), size=n_items, replace=False)
    correct = 0
    model.eval()

    with torch.no_grad():
        for idx in chosen:
            t = pool[idx]
            prompt = t[:prompt_chars]
            true_cont = t[prompt_chars:prompt_chars + cont_chars]

            negs = []
            while len(negs) < (K - 1):
                j = int(rng.integers(0, len(pool)))
                if j == idx:
                    continue
                tj = pool[j]
                negs.append(tj[prompt_chars:prompt_chars + cont_chars])

            cands = [true_cont] + negs
            rng.shuffle(cands)
            true_pos = cands.index(true_cont)

            scores = []
            for c in cands:
                full = prompt + c
                enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
                ids = enc["input_ids"][0].to(next(model.parameters()).device)

                p_len = len(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0])
                logits = model(ids.unsqueeze(0))
                logits = logits[:, :-1, :]
                labels = ids[1:].unsqueeze(0)

                start = max(0, p_len - 1)
                cont_logits = logits[:, start:, :]
                cont_labels = labels[:, start:]

                logp = F.log_softmax(cont_logits, dim=-1)
                token_logp = torch.gather(logp, 2, cont_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_logp.mean().item())

            pred = int(np.argmax(scores))
            if pred == true_pos:
                correct += 1

    return correct / n_items if n_items > 0 else 0.0


def train_eval_decoder(cfg, mixed_texts, human_eval, device, seed, ratio):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["decoder_model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = texts_to_lm_blocks(tok, mixed_texts, cfg["block_size"]).map(add_labels)
    eval_ds  = texts_to_lm_blocks(tok, human_eval,  cfg["block_size"]).map(add_labels)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["decoder_model_name"],
        load_in_4bit=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    run_name = f"decoder_ratio_{int(ratio*100)}_seed_{seed}"
    args = TrainingArguments(
        output_dir=f"./runs/{run_name}",
        run_name=run_name,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr_decoder"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        max_steps=cfg["max_steps"],
        fp16=(device == "cuda"),
        report_to="none",
        seed=seed
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
    trainer.train()
    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics["eval_loss"])
    ppl = perplexity_from_eval_loss(eval_loss)

    prompts = make_prompts(human_eval, n_prompts=cfg["gen_samples"], prompt_chars=cfg["prompt_chars"])
    gens = []
    trainer.model.eval()
    with torch.no_grad():
        for p in prompts:
            inputs = tok(p, return_tensors="pt", truncation=True, max_length=128).to(trainer.model.device)
            out = trainer.model.generate(
                **inputs,
                max_new_tokens=cfg["gen_max_new_tokens"],
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tok.eos_token_id
            )
            gens.append(tok.decode(out[0], skip_special_tokens=True))

    d1 = distinct_n(gens, n=1)
    d2 = distinct_n(gens, n=2)
    nsa = next_segment_selection_accuracy_causal(trainer.model, tok, human_eval, K=cfg["nextseg_k"], seed=seed)

    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"eval_loss_human": eval_loss, "perplexity_human": ppl, "distinct1": d1, "distinct2": d2, "nextseg_acc": nsa}


def train_eval_encdec(cfg, mixed_texts, human_eval, device, seed, ratio):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["encdec_model_name"])

    def make_pairs(texts, seed_local, n_pairs, in_chars=220, out_chars=220):
        rng = np.random.default_rng(seed_local)
        pool = [t for t in texts if len(t) >= (in_chars + out_chars + 50)]
        if len(pool) < n_pairs:
            n_pairs = max(100, len(pool))
        idxs = rng.choice(len(pool), size=n_pairs, replace=False)
        src, tgt = [], []
        for i in idxs:
            t = pool[i]
            src.append(t[:in_chars])
            tgt.append(t[in_chars:in_chars + out_chars])
        return src, tgt

    total_docs = cfg["docs_per_condition"]
    train_src, train_tgt = make_pairs(mixed_texts, seed, total_docs)
    eval_src,  eval_tgt  = make_pairs(human_eval, seed+1, min(400, len(human_eval)))

    train_ds = Dataset.from_dict({"src": train_src, "tgt": train_tgt})
    eval_ds  = Dataset.from_dict({"src": eval_src,  "tgt": eval_tgt})

    def preprocess(batch):
        model_inputs = tok(batch["src"], truncation=True, padding="max_length", max_length=256)
        with tok.as_target_tokenizer():
            labels = tok(batch["tgt"], truncation=True, padding="max_length", max_length=256)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["src","tgt"])
    eval_ds  = eval_ds.map(preprocess,  batched=True, remove_columns=["src","tgt"])

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["encdec_model_name"],
        load_in_8bit=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    run_name = f"encdec_ratio_{int(ratio*100)}_seed_{seed}"
    args = TrainingArguments(
        output_dir=f"./runs/{run_name}",
        run_name=run_name,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr_encdec"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        max_steps=cfg["max_steps"],
        fp16=(device == "cuda"),
        report_to="none",
        seed=seed
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
    trainer.train()
    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics["eval_loss"])
    ppl = perplexity_from_eval_loss(eval_loss)

    prompts = make_prompts(human_eval, n_prompts=cfg["gen_samples"], prompt_chars=cfg["prompt_chars"])
    gens = []
    trainer.model.eval()
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=256).to(trainer.model.device)
            out = trainer.model.generate(
                **inp,
                max_new_tokens=cfg["gen_max_new_tokens"],
                do_sample=True,
                temperature=0.9,
                top_p=0.95
            )
            gens.append(tok.decode(out[0], skip_special_tokens=True))

    d1 = distinct_n(gens, n=1)
    d2 = distinct_n(gens, n=2)

    def nsa_seq2seq(model, tokenizer, texts, K, seed_local):
        rng = np.random.default_rng(seed_local)
        pool = [t for t in texts if len(t) >= (cfg["prompt_chars"] + 250)]
        n_items = min(120, len(pool))
        if n_items < 10:
            return 0.0
        idxs = rng.choice(len(pool), size=n_items, replace=False)
        correct = 0
        model.eval()
        with torch.no_grad():
            for idx in idxs:
                t = pool[idx]
                src = t[:cfg["prompt_chars"]]
                true_tgt = t[cfg["prompt_chars"]:cfg["prompt_chars"] + 220]
                negs = []
                while len(negs) < (K - 1):
                    j = int(rng.integers(0, len(pool)))
                    if j == idx:
                        continue
                    tj = pool[j]
                    negs.append(tj[cfg["prompt_chars"]:cfg["prompt_chars"] + 220])
                cands = [true_tgt] + negs
                rng.shuffle(cands)
                true_pos = cands.index(true_tgt)
                scores = []
                for cand in cands:
                    inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(cand, return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
                    out = model(**inputs, labels=labels)
                    scores.append(-float(out.loss))
                pred = int(np.argmax(scores))
                if pred == true_pos:
                    correct += 1
        return correct / n_items

    nsa = nsa_seq2seq(trainer.model, tok, human_eval, cfg["nextseg_k"], seed)

    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"eval_loss_human": eval_loss, "perplexity_human": ppl, "distinct1": d1, "distinct2": d2, "nextseg_acc": nsa}


def train_eval_lstm(cfg, mixed_texts, human_eval, device, seed, ratio):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["decoder_model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = tok.vocab_size

    train_blocks = texts_to_lm_blocks(tok, mixed_texts, cfg["block_size"])
    eval_blocks  = texts_to_lm_blocks(tok, human_eval,  cfg["block_size"])

    model = LSTMLanguageModel(vocab_size, cfg["lstm_embed"], cfg["lstm_hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lstm_lr"])

    model.train()
    for _ in range(cfg["lstm_epochs"]):
        for xb, yb in make_torch_batches_from_blocks(train_blocks, cfg["batch_size"], device):
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()

    eval_loss = eval_lstm_loss(model, eval_blocks, cfg["batch_size"], device)
    ppl = perplexity_from_eval_loss(eval_loss)

    def lstm_generate(prompt, max_new):
        model.eval()
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
        ids = ids[:, -127:]
        for _ in range(max_new):
            logits = model(ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            ids = ids[:, -127:]
        return tok.decode(ids[0], skip_special_tokens=True)

    prompts = make_prompts(human_eval, n_prompts=cfg["gen_samples"], prompt_chars=cfg["prompt_chars"])
    gens = [lstm_generate(p, cfg["gen_max_new_tokens"]) for p in prompts]
    d1 = distinct_n(gens, n=1)
    d2 = distinct_n(gens, n=2)

    nsa = next_segment_selection_accuracy_lstm(model, tok, human_eval, K=cfg["nextseg_k"], seed=seed)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"eval_loss_human": eval_loss, "perplexity_human": ppl, "distinct1": d1, "distinct2": d2, "nextseg_acc": nsa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.json")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir_results"], exist_ok=True)
    os.makedirs(cfg["output_dir_artifacts"], exist_ok=True)

    # Load human data
    wiki = load_dataset("wikipedia/wikimedia", "20220301.en", split="train").shuffle(seed=42)
    human_train_raw = wiki.select(range(cfg["n_human_train_docs"]))["text"]
    human_eval_raw  = wiki.select(range(cfg["n_human_train_docs"], cfg["n_human_train_docs"] + cfg["n_human_eval_docs"]))["text"]

    human_train = clean_text_list(human_train_raw)
    human_eval  = clean_text_list(human_eval_raw)

    # Generate synthetic data with baseline
    gen_tok = AutoTokenizer.from_pretrained(cfg["synth_generator_name"])
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(cfg["synth_generator_name"]).to(device)
    generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tok, device=0 if device == "cuda" else -1)

    prompts = make_prompts(human_train, n_prompts=min(cfg["max_synth_samples"], len(human_train)), prompt_chars=200)
    synthetic = []
    for i, p in enumerate(prompts):
        out = generator(
            p,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=gen_tok.eos_token_id
        )[0]["generated_text"]
        synthetic.append(out)

    del gen_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    docs_per_condition = min(len(human_train), len(synthetic), cfg["n_human_train_docs"])
    cfg["docs_per_condition"] = docs_per_condition

    # Save manifest
    manifest = {
        "decoder_model": cfg["decoder_model_name"],
        "encdec_model": cfg["encdec_model_name"],
        "synth_generator": cfg["synth_generator_name"],
        "ratios": cfg["ratios"],
        "seeds": cfg["seeds"],
        "human_train_docs": len(human_train),
        "human_eval_docs": len(human_eval),
        "synthetic_docs": len(synthetic),
        "docs_per_condition": docs_per_condition,
        "block_size": cfg["block_size"]
    }
    with open(os.path.join(cfg["output_dir_artifacts"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Run experiments
    all_rows = []
    arches = [
        ("decoder_only_transformer", train_eval_decoder),
        ("encoder_decoder_transformer", train_eval_encdec),
        ("lstm_recurrent_baseline", train_eval_lstm)
    ]

    for arch_name, fn in arches:
        for ratio in cfg["ratios"]:
            for seed in cfg["seeds"]:
                mixed, n_h, n_s = make_mixed_texts(human_train, synthetic, ratio, docs_per_condition, seed)

                # Save small sample of mixed input for audit
                sample_path = os.path.join(cfg["output_dir_artifacts"], f"mix_{int(ratio*100)}_seed_{seed}.json")
                with open(sample_path, "w") as f:
                    json.dump({
                        "architecture": arch_name,
                        "ratio": ratio,
                        "seed": seed,
                        "human_docs": n_h,
                        "synth_docs": n_s,
                        "docs_sample_first_30": mixed[:30]
                    }, f, indent=2)

                metrics = fn(cfg, mixed, human_eval, device, seed, ratio)
                row = {
                    "architecture": arch_name,
                    "ratio_synth": ratio,
                    "seed": seed,
                    "human_docs": n_h,
                    "synth_docs": n_s,
                    **metrics
                }
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(os.path.join(cfg["output_dir_results"], "results_full.csv"), index=False)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(cfg["output_dir_results"], "results_full.csv"), index=False)

    summary_df = (
        results_df
        .groupby(["architecture", "ratio_synth"])
        .agg({
            "perplexity_human": ["mean", "std"],
            "eval_loss_human": ["mean", "std"],
            "distinct1": ["mean", "std"],
            "distinct2": ["mean", "std"],
            "nextseg_acc": ["mean", "std"]
        })
        .reset_index()
    )
    summary_df.columns = ["_".join([c for c in col if c]) for col in summary_df.columns.values]
    summary_df.to_csv(os.path.join(cfg["output_dir_results"], "results_summary.csv"), index=False)

    # Stats
    stats_lines = []
    try:
        results_df["architecture"] = results_df["architecture"].astype("category")
        m1 = smf.ols("perplexity_human ~ ratio_synth * architecture", data=results_df).fit()
        stats_lines.append("=== OLS: perplexity_human ~ ratio_synth * architecture ===\n")
        stats_lines.append(m1.summary().as_text())
        stats_lines.append("\n\n")
        m2 = smf.ols("nextseg_acc ~ ratio_synth * architecture", data=results_df).fit()
        stats_lines.append("=== OLS: nextseg_acc ~ ratio_synth * architecture ===\n")
        stats_lines.append(m2.summary().as_text())
    except Exception as e:
        stats_lines.append(f"Stats failed: {repr(e)}")

    with open(os.path.join(cfg["output_dir_results"], "stats_regression.txt"), "w") as f:
        f.write("\n".join(stats_lines))


if __name__ == "__main__":
    main()
