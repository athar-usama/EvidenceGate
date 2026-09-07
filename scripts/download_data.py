"""Downloads the IDRiD segmentation subset and reorganizes it under data/raw/.

Source: IDRiD (Porwal et al., 2018, CC BY 4.0), redistributed as a directly
downloadable zip on Zenodo (record 17219542) since the original IEEE DataPort
host requires an account.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/api/records/17219542/files/A.%20Segmentation.zip/content"
EXPECTED_SIZE = 584_315_841

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ZIP = REPO_ROOT / "data" / "cache" / "segmentation.zip"
RAW_DIR = REPO_ROOT / "data" / "raw"

CATEGORY_TO_CODE = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
    "5. Optic Disc": "OD",
}
SPLIT_DIR_NAMES = {"train": "a. Training Set", "test": "b. Testing Set"}


def download(force: bool = False) -> None:
    CACHE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_ZIP.exists() and CACHE_ZIP.stat().st_size == EXPECTED_SIZE and not force:
        print(f"Zip already present at {CACHE_ZIP}, skipping download.")
        return

    with requests.get(ZENODO_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", EXPECTED_SIZE))
        with open(CACHE_ZIP, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="IDRiD segmentation") as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def reorganize() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"{RAW_DIR} already populated, skipping extraction.")
        return

    (RAW_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "images" / "test").mkdir(parents=True, exist_ok=True)
    for code in CATEGORY_TO_CODE.values():
        (RAW_DIR / "masks" / code / "train").mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "masks" / code / "test").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(CACHE_ZIP) as z:
        for split, split_dir_name in SPLIT_DIR_NAMES.items():
            image_prefix = f"A. Segmentation/1. Original Images/{split_dir_name}/"
            for name in z.namelist():
                if name.startswith(image_prefix) and name.lower().endswith(".jpg"):
                    dest = RAW_DIR / "images" / split / Path(name).name
                    with z.open(name) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)

            for category_name, code in CATEGORY_TO_CODE.items():
                mask_prefix = f"A. Segmentation/2. All Segmentation Groundtruths/{split_dir_name}/{category_name}/"
                for name in z.namelist():
                    if name.startswith(mask_prefix) and name.lower().endswith(".tif"):
                        dest = RAW_DIR / "masks" / code / split / Path(name).name
                        with z.open(name) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)

    n_train = len(list((RAW_DIR / "images" / "train").glob("*.jpg")))
    n_test = len(list((RAW_DIR / "images" / "test").glob("*.jpg")))
    print(f"Reorganized {n_train} train / {n_test} test images into {RAW_DIR}")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    reorganize()
