"""Download the WNUT-17 dataset files into data/raw/."""

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/"
FILES = ["wnut17train.conll", "emerging.dev.conll", "emerging.test.annotated"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download() -> None:
    """Fetch each raw dataset file if it is not already present locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = DATA_DIR / filename
        if destination.exists():
            continue
        urllib.request.urlretrieve(BASE_URL + filename, destination)


if __name__ == "__main__":
    download()
