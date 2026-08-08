#!/bin/bash

# Data fetch instructions and downloads setup
echo "=========================================================="
echo "Cardiovascular Disease Prediction Dataset Downloader"
echo "=========================================================="

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "Note: Access to the MIMIC-III Clinical Database requires credentialed access."
echo "Please register on PhysioNet (https://physionet.org) and complete the CITI training."
echo "Once credentialed, download patient records and place mimic_patients.csv in: $DATA_DIR"
echo ""
echo "Generating local synthetic data mockups automatically for local execution..."
python -c "
from src.utils.config import Config
from src.data.load_dataset import load_or_create_datasets
config = Config()
load_or_create_datasets(config)
"

echo "Setup done. Data generated in data/raw/"
