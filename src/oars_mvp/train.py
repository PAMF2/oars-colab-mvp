import argparse
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from .dataset import SyntheticProofDataset
from .model import OARSMVP


@dataclass
class Metrics:
    loss: float
    acc: float
    reward: float
    allocator_entropy: float


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * (probs + eps).log()).sum(dim=-1).mean()


def evaluate(model, loader, device, mode):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ent_vals = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits, b_probs, c_probs = model(x, mode=mode)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_n += y.numel()
            total_loss += loss.item() * y.size(0)

            if mode.startswith("arla"):
                ent = entropy_from_probs(b_probs) + entropy_from_probs(c_probs.flatten(0, 1))
                ent_vals.append(ent.item())

    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)
    reward = acc
    entropy = float(sum(ent_vals) / len(ent_vals)) if ent_vals else 0.0
    return Metrics(loss=avg_loss, acc=acc, reward=reward, allocator_entropy=entropy)


def run_experiment(cfg: dict) -> dict:
    device = "cuda" if (cfg.get("device", "auto") == "auto" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(cfg["seed"])

    dataset = SyntheticProofDataset(
        n_samples=cfg["samples"],
        input_dim=cfg["input_dim"],
        num_blocks=cfg["num_blocks"],
        seed=cfg["seed"],
    )
    n_train = int(cfg["samples"] * cfg["train_split"])
    n_val = cfg["samples"] - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = OARSMVP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        concepts_per_block=cfg["concepts_per_block"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    t0 = time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits, _, _ = model(x, mode=cfg["mode"])
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    metrics = evaluate(model, val_loader, device=device, mode=cfg["mode"])
    elapsed = time.time() - t0

    result = {
        "mode": cfg["mode"],
        "seed": cfg["seed"],
        "samples": cfg["samples"],
        "epochs": cfg["epochs"],
        "metrics": metrics.__dict__,
        "device": device,
        "runtime_sec": round(elapsed, 3),
    }

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"result_{cfg['mode']}_seed{cfg['seed']}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.samples is not None:
        cfg["samples"] = args.samples

    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
