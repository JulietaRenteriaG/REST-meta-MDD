import re
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch


SLICES_DIR = Path("outputs/slices")

# ReHoMap_S9-1-0049 → grupo 1 = MDD, grupo 2 = HC
LABEL_RE = re.compile(r"S\d+-(\d+)-\d+")


def parse_label(filename: str) -> int:
    """Extrae label del nombre: grupo 1 → MDD=1, grupo 2 → HC=0."""
    m = LABEL_RE.search(filename)
    if m is None:
        raise ValueError(f"No se pudo extraer label de: {filename}")
    grupo = int(m.group(1))
    if grupo == 1:
        return 1  # MDD
    elif grupo == 2:
        return 0  # HC
    else:
        raise ValueError(f"Grupo inesperado {grupo} en: {filename}")


class ReHoDataset(Dataset):
    def __init__(self, slices_dir: Path = SLICES_DIR, transform=None):
        self.files = sorted(slices_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No se encontraron .npy en {slices_dir}")

        self.labels    = [parse_label(f.stem) for f in self.files]
        self.transform = transform

        mdd = sum(self.labels)
        hc  = len(self.labels) - mdd
        print(f"Dataset cargado: {len(self.files)} sujetos  |  MDD={mdd}  HC={hc}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)  # (3, 64, 64)
        y = self.labels[idx]

        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y

    def site_ids(self):
        """Devuelve lista de site IDs (ej. 'S1', 'S9') para leave-one-site-out CV."""
        return [re.search(r"(S\d+)-", f.stem).group(1) for f in self.files]


def get_loaders(batch_size=16, val_split=0.2, seed=42):
    """Split train/val aleatorio. Para producción usar leave-one-site-out."""
    from torch.utils.data import random_split

    ds = ReHoDataset()
    n_val   = int(len(ds) * val_split)
    n_train = len(ds) - n_val

    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {n_train}  Val: {n_val}")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_loaders()
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}  Labels: {y}")