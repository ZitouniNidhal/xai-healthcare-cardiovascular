#!/bin/bash

# Setup script for XAI Healthcare Cardiovascular Diagnostics environment
echo "=========================================================="
echo "Setting up Conda Environment for XAI Cardiovascular Project"
echo "=========================================================="

# Check if conda command exists
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Installing dependencies via Pip in virtualenv."
    python -m venv venv
    source venv/Scripts/activate || source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Conda found. Creating virtual environment 'xai-cardio'..."
    conda create -n xai-cardio python=3.10 -y
    
    # Initialize conda in bash shell
    eval "$(conda shell.bash hook)"
    conda activate xai-cardio
    
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo "Environment setup complete! Run 'conda activate xai-cardio' to start."
