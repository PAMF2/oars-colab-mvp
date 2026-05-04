import argparse
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from .dataset import MiniF2FLikeDataset, SyntheticProofDataset
from .model import OARSMVP


@dataclass
class Metrics:
    loss: float
    acc: float
    reward: float
    allocator_entropy: float
    f1_macro: float


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * (probs + eps).log()).sum(dim=-1).mean()


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum().item()
        fp = ((y_true != c) & (y_pred == c)).sum().item()
        fn = ((y_true == c) & (y_pred != c)).sum().item()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / max(len(f1s), 1))


def evaluate(model, loader, device, mode, task_type: str, num_classes: int):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ent_vals = []
    yt, yp = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            if task_type == "multiclass":
                y = batch["class_label"].to(device)
            else:
                y = batch["y"].to(device)

            logits, b_probs, c_probs = model(x, mode=mode)

            if task_type == "multiclass":
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=-1)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
                preds = (torch.sigmoid(logits) > 0.5).float()

            total_correct += (preds == y).sum().item()
            total_n += y.numel()
            total_loss += loss.item() * y.size(0)
            yt.append(y.detach().cpu())
            yp.append(preds.detach().cpu())

            if mode.startswith("arla"):
                ent = entropy_from_probs(b_probs) + entropy_from_probs(c_probs.flatten(0, 1))
                ent_vals.append(ent.item())

    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)
    reward = acc
    entropy = float(sum(ent_vals) / len(ent_vals)) if ent_vals else 0.0

    y_true = torch.cat(yt) if yt else torch.tensor([])
    y_pred = torch.cat(yp) if yp else torch.tensor([])
    if task_type == "multiclass":
        f1 = macro_f1(y_true, y_pred, num_classes=num_classes)
    else:
        f1 = macro_f1(y_true.long(), y_pred.long(), num_classes=2)

    return Metrics(loss=avg_loss, acc=acc, reward=reward, allocator_entropy=entropy, f1_macro=f1)


def maybe_init_wandb(cfg: dict):
    if not cfg.get("wandb", {}).get("enabled", False):
        return None
    try:
        import wandb

        run = wandb.init(
            project=cfg["wandb"].get("project", "oars-colab-mvp"),
            name=cfg["wandb"].get("run_name"),
            config=cfg,
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"[warn] wandb init failed: {e}")
        return None


def _build_loader_from_path(path: str, cfg: dict, shuffle: bool):
    ds = MiniF2FLikeDataset(path=path, input_dim=cfg["input_dim"], num_blocks=cfg["num_blocks"])
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle), len(ds)


def build_dataloaders(cfg: dict):
    ds_cfg = cfg.get("dataset", {})
    ds_type = ds_cfg.get("type", "synthetic")

    if ds_type == "synthetic":
        dataset = SyntheticProofDataset(
            n_samples=cfg["samples"],
            input_dim=cfg["input_dim"],
            num_blocks=cfg["num_blocks"],
            seed=cfg["seed"],
        )
        total_samples = len(dataset)
        n_train = int(total_samples * cfg["train_split"])
        n_val = total_samples - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"]))
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
        return train_loader, val_loader, val_loader, {"train": n_train, "val": n_val, "test": n_val}

    if ds_type == "minif2f_like":
        train_path = ds_cfg.get("train_path")
        val_path = ds_cfg.get("val_path")
        test_path = ds_cfg.get("test_path")
        if train_path and val_path and test_path:
            train_loader, n_train = _build_loader_from_path(train_path, cfg, shuffle=True)
            val_loader, n_val = _build_loader_from_path(val_path, cfg, shuffle=False)
            test_loader, n_test = _build_loader_from_path(test_path, cfg, shuffle=False)
            return train_loader, val_loader, test_loader, {"train": n_train, "val": n_val, "test": n_test}

        dataset = MiniF2FLikeDataset(path=ds_cfg["path"], input_dim=cfg["input_dim"], num_blocks=cfg["num_blocks"])
        total_samples = len(dataset)
        n_train = int(total_samples * cfg.get("train_split", 0.8))
        n_val = total_samples - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"]))
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
        return train_loader, val_loader, val_loader, {"train": n_train, "val": n_val, "test": n_val}

    raise ValueError(f"Unknown dataset.type: {ds_type}")


def run_experiment(cfg: dict) -> dict:
    device = "cuda" if (cfg.get("device", "auto") == "auto" and torch.cuda.is_available()) else "cpu"
    task_type = cfg.get("task", {}).get("type", "multiclass")
    num_classes = int(cfg.get("task", {}).get("num_classes", 4)) if task_type == "multiclass" else 1
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader, test_loader, sizes = build_dataloaders(cfg)

    model = OARSMVP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        concepts_per_block=cfg["concepts_per_block"],
        num_classes=num_classes,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    wandb_run = maybe_init_wandb(cfg)

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            if task_type == "multiclass":
                y = batch["class_label"].to(device)
            else:
                y = batch["y"].to(device)

            logits, _, _ = model(x, mode=cfg["mode"])
            loss = F.cross_entropy(logits, y) if task_type == "multiclass" else F.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device=device, mode=cfg["mode"], task_type=task_type, num_classes=max(num_classes, 2))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": epoch_loss / max(len(train_loader), 1),
                    "val/acc": val_metrics.acc,
                    "val/f1_macro": val_metrics.f1_macro,
                    "epoch": epoch + 1,
                }
            )

    metrics = evaluate(model, test_loader, device=device, mode=cfg["mode"], task_type=task_type, num_classes=max(num_classes, 2))
    elapsed = time.time() - t0

    result = {
        "mode": cfg["mode"],
        "seed": cfg["seed"],
        "samples": sizes,
        "epochs": cfg["epochs"],
        "dataset_type": cfg.get("dataset", {}).get("type", "synthetic"),
        "task_type": task_type,
        "metrics": metrics.__dict__,
        "device": device,
        "runtime_sec": round(elapsed, 3),
    }

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"result_{cfg['mode']}_seed{cfg['seed']}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in result["metrics"].items()})
        wandb_run.log({"eval/runtime_sec": result["runtime_sec"]})
        wandb_run.finish()

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
