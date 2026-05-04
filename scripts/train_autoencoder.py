import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from oars_mvp.dataset import read_jsonl
from oars_mvp.autoencoder import FeatureAutoencoder


class FeatureDataset(Dataset):
    def __init__(self, rows):
        self.x = [r["features"] for r in rows]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = model.reconstruction_loss(x, x_hat)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/minif2f_prepared.jsonl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/autoencoder")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    rows = read_jsonl(args.data)
    if not rows:
        raise RuntimeError("empty dataset")
    input_dim = len(rows[0]["features"])

    ds = FeatureDataset(rows)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_autoencoder.pt"

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = model.reconstruction_loss(x, x_hat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                },
                ckpt,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    report = {
        "data": args.data,
        "device": device,
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "checkpoint": str(ckpt),
        "history": history,
    }

    (out_dir / "autoencoder_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_loss": best_val, "checkpoint": str(ckpt)}, indent=2))


if __name__ == "__main__":
    main()
