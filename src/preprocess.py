import os
import numpy as np
import nibabel as nib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW   = Path(os.getenv("RAW_DATA_PATH"))  # apunta a Results/
OUT   = Path("outputs/slices")
OUT.mkdir(parents=True, exist_ok=True)

REHO_DIR = "ReHo_FunImgARglobalCWF"
TARGET   = (64, 64)               # tamaño al que se redimensionan los slices


def load_volume(path: Path) -> np.ndarray:
    """Carga un .nii o .nii.gz y devuelve array float32."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def zscore(vol: np.ndarray) -> np.ndarray:
    """Z-score por volumen completo (excluye voxels fuera de máscara ~0)."""
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    mu, sd = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mu) / (sd + 1e-8)
    return out


def resize_slice(s: np.ndarray, size: tuple) -> np.ndarray:
    """Redimensiona un slice 2D con interpolacion bilineal simple."""
    from scipy.ndimage import zoom
    factors = (size[0] / s.shape[0], size[1] / s.shape[1])
    return zoom(s, factors, order=1)


def extract_25d(vol: np.ndarray) -> np.ndarray:
    """
    Extrae 3 slices ortogonales del centro del volumen.
    Devuelve array (3, H, W) normalizado a TARGET.
    """
    cx, cy, cz = [d // 2 for d in vol.shape]
    axial    = resize_slice(vol[:, :, cz], TARGET)   # plano axial
    coronal  = resize_slice(vol[:, cy, :], TARGET)   # plano coronal
    sagital  = resize_slice(vol[cx, :, :], TARGET)   # plano sagital
    return np.stack([axial, coronal, sagital], axis=0)  # (3, 64, 64)


def process_all():
    """
    Estructura real del dataset:
    RAW / ReHo_FunImgARglobalCWF / SubjectFolder / SubjectFolder.nii
    """
    reho_root = RAW / REHO_DIR  # Results/ReHo_FunImgARglobalCWF/
    print(f"Buscando en: {reho_root}")
    print(f"Existe: {reho_root.exists()}")
    print(f"Contenido: {list(reho_root.iterdir())[:5]}")
    
    if not reho_root.exists():
        print(f"[ERROR] No se encontró: {reho_root}")
        return

    nii_files = sorted(reho_root.glob("*.nii.gz")) + sorted(reho_root.glob("*.nii"))
    print(f"Sujetos encontrados: {len(nii_files)}")
    OUT.mkdir(parents=True, exist_ok=True)

    ok, err, skip = 0, 0, 0
    for nii in nii_files:
        subj_id  = nii.name.replace(".nii.gz", "").replace(".nii", "")
        out_file = OUT / f"{subj_id}.npy"

        if out_file.exists():
            skip += 1
            continue

        try:
            vol   = load_volume(nii)
            vol   = zscore(vol)
            patch = extract_25d(vol)
            np.save(out_file, patch)
            print(f"  OK  {subj_id}  shape={patch.shape}")
            ok += 1
        except Exception as e:
            print(f"  ERR {subj_id}: {e}")
            err += 1

    print(f"\nCompletado — OK:{ok}  ERR:{err}  Skip:{skip}")
    print(f"Slices en: {OUT.resolve()}")


def main():
    process_all()


if __name__ == "__main__":
    main()