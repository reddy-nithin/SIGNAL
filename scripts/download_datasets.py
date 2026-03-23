"""
SIGNAL — Day 1 Dataset Downloader
===================================
Downloads 6 datasets from HuggingFace to datasets/.
Run from project root: python scripts/download_datasets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

DATASET_MAP = [
    {
        "name": "RMHD (Reddit Mental Health Posts)",
        "hf_id": "solomonk/reddit_mental_health_posts",
        "save_dir": "reddit_mh_rmhd",
        "split": "train",
        "description": "Reddit posts from mental health subreddits with subreddit labels",
    },
    {
        "name": "Reddit MH Labeled (Classification)",
        "hf_id": "kamruzzaman-asif/reddit-mental-health-classification",
        "save_dir": "reddit_mh_labeled",
        "split": "train",
        "description": "Reddit mental health posts with classification labels",
    },
    {
        "name": "Mental Health Reddit (Cleaned)",
        "hf_id": "fadodr/mental_health_cleaned_dataset",
        "save_dir": "reddit_mh_cleaned",
        "split": "train",
        "description": "Mental health counseling dialogues (cleaned)",
    },
    {
        "name": "Reddit MH Cleaned Research",
        "hf_id": "hugginglearners/reddit-depression-cleaned",
        "save_dir": "reddit_mh_research",
        "split": "train",
        "description": "Reddit depression posts (binary labeled, cleaned)",
    },
    {
        "name": "UCI Drug Review",
        "hf_id": "lewtun/drug-reviews",
        "save_dir": "uci_drug_reviews",
        "split": "train",
        "description": "UCI drug review dataset with drug names, conditions, and patient ratings",
    },
    {
        "name": "DepressionEmo (Reddit Depression Cleaned)",
        "hf_id": "mrjunos/depression-reddit-cleaned",
        "save_dir": "depression_emo",
        "split": "train",
        "description": "Reddit depression posts with emotion/severity labels",
    },
]


def download_dataset(entry: dict) -> dict:
    """Download a single dataset and save as CSV. Returns audit info."""
    save_path = DATASETS_DIR / entry["save_dir"]
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "data.csv"

    print(f"\n{'='*60}")
    print(f"Downloading: {entry['name']}")
    print(f"  HuggingFace ID : {entry['hf_id']}")
    print(f"  Save dir       : {save_path}")

    try:
        ds = load_dataset(entry["hf_id"], trust_remote_code=True)

        # Use the requested split, fallback to first available
        split = entry["split"]
        if split not in ds:
            split = list(ds.keys())[0]
            print(f"  (split '{entry['split']}' not found, using '{split}')")

        df = ds[split].to_pandas()
        df.to_csv(csv_path, index=False)

        print(f"  Rows           : {len(df):,}")
        print(f"  Columns        : {list(df.columns)}")
        print(f"  Saved to       : {csv_path}")
        print(f"  Status         : OK")

        return {
            "name": entry["name"],
            "hf_id": entry["hf_id"],
            "save_dir": entry["save_dir"],
            "rows": len(df),
            "columns": list(df.columns),
            "status": "OK",
            "csv_path": str(csv_path),
        }

    except Exception as e:
        print(f"  Status         : FAILED — {e}")
        return {
            "name": entry["name"],
            "hf_id": entry["hf_id"],
            "save_dir": entry["save_dir"],
            "rows": 0,
            "columns": [],
            "status": f"FAILED: {e}",
            "csv_path": None,
        }


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(DATASET_MAP)} datasets to {DATASETS_DIR}")

    results = []
    for entry in DATASET_MAP:
        results.append(download_dataset(entry))

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    total_rows = 0
    for r in results:
        status = "✓" if r["status"] == "OK" else "✗"
        print(f"{status} {r['name']}: {r['rows']:,} rows — {r['status']}")
        total_rows += r["rows"]

    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} datasets downloaded | Total rows: {total_rows:,}")
    return results


if __name__ == "__main__":
    main()
