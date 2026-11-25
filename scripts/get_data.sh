#!/bin/bash
# Prepares the Avazu training data in data/raw/.
#
# Getting train.gz (needs a Kaggle account with the competition rules accepted):
#   Browser: https://www.kaggle.com/competitions/avazu-ctr-prediction/data
#            download train.gz and move it to data/raw/train.gz
#   CLI:     kaggle competitions download -c avazu-ctr-prediction -f train.gz -p data/raw
#            (the API token lives in ~/.kaggle/, never in this repo)
#
# This script does not download anything. It checks train.gz and decompresses it
# to train.csv, keeping the .gz. Spark cannot split a gzip file, so it would read
# a .gz in one task on one core; a plain CSV is read in parallel chunks.
set -euo pipefail

RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
GZ="$RAW_DIR/train.gz"
CSV="$RAW_DIR/train.csv"

mkdir -p "$RAW_DIR"

if [[ ! -f "$GZ" ]]; then
    echo "Missing $GZ. Download it as described at the top of this script." >&2
    exit 1
fi

# A download that arrived as a zip archive needs unzipping first.
if [[ "$(head -c 2 "$GZ" | od -An -tx1 | tr -d ' ')" != "1f8b" ]]; then
    echo "$GZ is not a gzip file. If Kaggle gave you a .zip, unzip it first." >&2
    exit 1
fi

if [[ -f "$CSV" ]]; then
    echo "$CSV already exists, leaving it as is."
else
    echo "Decompressing $GZ (about 6 GB of CSV, takes a minute or two)..."
    # Write to a temporary name first so an interrupted run never leaves a
    # truncated train.csv that looks complete.
    gzip -dc "$GZ" > "$CSV.partial"
    mv "$CSV.partial" "$CSV"
fi

echo "Header: $(head -n 1 "$CSV")"
echo "Size:   $(du -h "$CSV" | cut -f1)"
