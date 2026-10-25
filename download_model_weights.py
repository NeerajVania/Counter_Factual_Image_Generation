import argparse
import zipfile
from pathlib import Path
import requests

ZENODO_ID = "18781460"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_ID}/files"

CORE_MODELS = ["VAE", "ENC"]

MODEL_VARIANTS = [
    "base",
    "basestruct",
    "equi",
    "equistruct",
    "learn",
    "learnstruct",
]

def download(filename: str, dest: Path):
    url = f"{ZENODO_BASE_URL}/{filename}?download=1"
    print(f"Downloading from {url} → {dest} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")


def extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted {zip_path} → {out_dir}")
    zip_path.unlink()


def main(selected_model: str):
    base_dir = Path("model_weights")
    diffusion_dir = base_dir / "diffusion"
    base_dir.mkdir(exist_ok=True)
    diffusion_dir.mkdir(parents=True, exist_ok=True)

    if selected_model == "all":
        models_to_download = MODEL_VARIANTS
    else:
        if selected_model not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model '{selected_model}'. "
                f"Choose from {MODEL_VARIANTS} or 'all'."
            )
        models_to_download = [selected_model]

    for model in CORE_MODELS:
        zip_name = f"{model}.zip"
        zip_path = Path(zip_name)

        download(zip_name, zip_path)
        extract_zip(zip_path, base_dir / model)
    
    for model in models_to_download:
        zip_name = f"{model}.zip"
        zip_path = Path(zip_name)

        download(zip_name, zip_path)
        extract_zip(zip_path, diffusion_dir / model)

    print("Done. model_weights directory is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract model weights from Zenodo.")
    parser.add_argument(
        "--model",
        default="base",
        help="Model variant to download "
             "(base, basestruct, equi, equistruct, learn, learnstruct, all)",
    )    
    args = parser.parse_args()
    main(selected_model=args.model)