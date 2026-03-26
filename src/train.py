import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from dataset import ReHoDataset
from model import ReHoCNN

# ── configuración ─────────────────────────────────────────────────────────────
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-3    # menos agresivo que antes
PATIENCE     = 12
OUT_DIR      = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) < 0.5:
        x = x + torch.randn_like(x) * 0.02
    if torch.rand(1) < 0.5:
        x = x * (0.95 + torch.rand(1) * 0.10)
    return x


def make_loaders_loso(ds, val_site: str, batch_size: int):
    """Divide train/val por sitio — val_site queda completamente fuera del train."""
    sites  = ds.site_ids()
    tr_idx = [i for i, s in enumerate(sites) if s != val_site]
    vl_idx = [i for i, s in enumerate(sites) if s == val_site]

    train_ds = Subset(ds, tr_idx)
    val_ds   = Subset(ds, vl_idx)

    # WeightedRandomSampler para desbalance en train
    tr_labels = [ds.labels[i] for i in tr_idx]
    counts    = np.bincount(tr_labels)
    weights   = 1.0 / counts[tr_labels]
    sampler   = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    vl_mdd = sum(ds.labels[i] for i in vl_idx)
    vl_hc  = len(vl_idx) - vl_mdd
    print(f"  Val site {val_site}: {len(vl_idx)} sujetos  MDD={vl_mdd}  HC={vl_hc}")
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x    = augment(x)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (model(x).argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n     = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(1)
        n     += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    bacc = balanced_accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return total_loss / n, acc, bacc, f1, auc, all_preds, all_labels


def run_fold(ds, val_site: str, device):
    print(f"\n{'='*50}")
    print(f"Fold: val_site = {val_site}")

    train_loader, val_loader = make_loaders_loso(ds, val_site, BATCH_SIZE)
    model     = ReHoCNN(dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc, best_bacc, best_auc, best_f1 = 0.0, 0.0, 0.0, 0.0
    patience_cnt = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc               = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_bacc, vl_f1, vl_auc, preds, lbs = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:02d}  train_acc={tr_acc:.3f}  "
              f"val_acc={vl_acc:.3f}  bacc={vl_bacc:.3f}  "
              f"f1={vl_f1:.3f}  auc={vl_auc:.3f}")

        if vl_bacc > best_bacc:   # usamos balanced_acc como criterio de guardado
            best_acc, best_bacc, best_auc, best_f1 = vl_acc, vl_bacc, vl_auc, vl_f1
            patience_cnt  = 0
            best_preds    = preds
            best_labels   = lbs
            torch.save(model.state_dict(), OUT_DIR / f"best_{val_site}.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping en época {epoch}")
                break

    return {"acc": best_acc, "bacc": best_bacc, "auc": best_auc, "f1": best_f1,
            "preds": best_preds, "labels": best_labels}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    ds    = ReHoDataset()
    sites = sorted(set(ds.site_ids()))
    print(f"Sitios encontrados: {len(sites)} → {sites}")

    results = {}
    for site in sites[:6]:   # cambia a sites para correr todos
        result = run_fold(ds, site, device)
        results[site] = result
        print(f"  Mejor {site}: acc={result['acc']:.3f}  bacc={result['bacc']:.3f}  auc={result['auc']:.3f}  f1={result['f1']:.3f}")

    print(f"\n{'='*50}")
    print("Resumen LOSO:")
    for site, v in results.items():
        print(f"  {site}: acc={v['acc']:.3f}  bacc={v['bacc']:.3f}  auc={v['auc']:.3f}  f1={v['f1']:.3f}")
    print(f"\nMedia AUC:  {np.mean([v['auc']  for v in results.values()]):.3f}")
    print(f"Media BACC: {np.mean([v['bacc'] for v in results.values()]):.3f}")


if __name__ == "__main__":
    main()