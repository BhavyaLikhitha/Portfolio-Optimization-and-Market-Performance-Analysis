from pathlib import Path
import shutil

import kagglehub


DATASET = "darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily"
TARGET_DIR = Path("data/local")
FILES_TO_COPY = {
    "sp500_stocks.csv": "sp500_stocks.csv",
    "sp500_companies.csv": "sp500_companies.csv",
}


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded dataset cache path: {dataset_path}")

    copied = []
    for source_name, target_name in FILES_TO_COPY.items():
        source_path = dataset_path / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Expected file not found in Kaggle dataset cache: {source_path}")

        target_path = TARGET_DIR / target_name
        shutil.copy2(source_path, target_path)
        copied.append(target_path)

    print("Copied files:")
    for path in copied:
        print(f" - {path}")


if __name__ == "__main__":
    main()
