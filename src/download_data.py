"""Download the Telco Customer Churn CSV into data/raw/ if it is not there yet.

The file hash is checked so the analysis always runs on the same data.
"""

import hashlib
import urllib.request

from config import DATA_SHA256, DATA_URL, RAW_DIR, RAW_FILE


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f"Raw file already present: {RAW_FILE.name}")
    else:
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW_FILE)
        print(f"Saved {RAW_FILE.name} ({RAW_FILE.stat().st_size:,} bytes)")

    actual = file_sha256(RAW_FILE)
    if actual != DATA_SHA256:
        raise ValueError(
            f"{RAW_FILE.name} does not match the expected file.\n"
            f"expected sha256: {DATA_SHA256}\nactual sha256:   {actual}\n"
            "Delete the file and run again, or see data/README.md."
        )
    print("File hash matches the expected dataset.")


if __name__ == "__main__":
    download_data()
