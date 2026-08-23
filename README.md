# 🫀 XAI-Healthcare: Explainable AI for Early Cardiovascular Disease Detection

[![CI](https://github.com/ZitouniNidhal/xai-healthcare-cardiovascular/actions/workflows/ci.yml/badge.svg)](https://github.com/ZitouniNidhal/xai-healthcare-cardiovascular/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

Cardiovascular diseases (CVDs) are the **#1 cause of death globally** (WHO, 2023). This project develops an **Explainable AI (XAI) system** that analyzes medical data — electrocardiograms (ECG), cardiac imaging, and patient history — to predict CVD risks while providing **clear, actionable explanations** to clinicians.

### Key Features

- **Hybrid AI Model**: Combines XGBoost/Random Forest with lightweight neural networks for structured and unstructured data
- **Multi-modal Explainability**:
  - **SHAP** (SHapley Additive exPlanations) for structured feature contributions
  - **LIME** (Local Interpretable Model-agnostic Explanations) for tabular data
  - **Grad-CAM** for medical image visualization (cardiac MRI/CT scans)
- **Real-time Explanations**: Contribution charts, heatmaps, and natural language explanations
- **Clinical Explanation Card**: Every patient report includes risk bands, leading risk/protective drivers, and agreement between tabular and ECG models
- **Clinical Dashboard**: Interactive interface for physicians with HL7/FHIR compatibility
- **Clinical Validation**: Benchmarked on MIMIC-III and PhysioNet datasets

## 🏗️ Architecture

```
Patient Data (ECG, MRI, Clinical Records)
        │
        ▼
┌─────────────────────┐
│   Data Preprocessing │ ← Normalization, Feature Engineering, Augmentation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Hybrid AI Model    │ ← XGBoost + CNN (ECG) + Image Analysis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   XAI Module         │ ← SHAP + LIME + Grad-CAM
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Clinical Dashboard │ ← Interactive Visualizations + Reports
└─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Conda or pip
- CUDA-compatible GPU (optional, for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/ZitouniNidhal/xai-healthcare-cardiovascular.git
cd xai-healthcare-cardiovascular

# Create virtual environment
conda create -n xai-cardio python=3.10
conda activate xai-cardio

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
bash scripts/setup_environment.sh
```

### Download Data

```bash
# Download public datasets (MIMIC-III requires credentialed access)
bash scripts/download_data.sh
```

### Run the Pipeline

```bash
# Run the complete pipeline (preprocessing → training → explainability)
python src/main.py

# Or run individual components
python src/main.py --stage preprocess
python src/main.py --stage train
python src/main.py --stage explain
```

### Configuration

Edit the configuration files in `configs/`:
- `params.yaml` — Model hyperparameters
- `paths.yaml` — Data and output paths

## 📊 Results

| Model | AUC-ROC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost + SHAP | 0.94 | 0.89 | 0.91 | 0.87 |
| CNN (ECG) | 0.92 | 0.86 | 0.88 | 0.84 |
| Hybrid Model | **0.96** | **0.91** | **0.93** | **0.90** |

> See [docs/results.md](docs/results.md) for detailed analysis and visualizations.

## 📁 Project Structure

```
xai-healthcare-cardiovascular/
├── docs/               # Documentation and publications
├── src/                # Source code (data, models, explainability, utils)
├── notebooks/          # Jupyter notebooks for exploration
├── experiments/        # Logs, saved models, and outputs
├── data/               # Raw and processed datasets (not committed)
├── tests/              # Unit and integration tests
├── configs/            # YAML configuration files
├── scripts/            # Setup, download, and deployment scripts
└── .github/            # CI/CD workflows
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
pytest tests/test_explainability.py -v
```

## 📖 Documentation

- [Project Proposal](docs/project_proposal.md)
- [Methodology](docs/methodology.md)
- [Results](docs/results.md)
- [API Documentation](docs/api_documentation.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- **Author**: Nidhal Zitouni
- **GitHub**: [ZitouniNidhal](https://github.com/ZitouniNidhal)

## 🙏 Acknowledgments

- [MIMIC-III](https://mimic.mit.edu/) — MIT Laboratory for Computational Physiology
- [PhysioNet](https://physionet.org/) — Research Resource for Complex Physiologic Signals
- [SHAP](https://github.com/slundberg/shap) — Scott Lundberg
- [LIME](https://github.com/marcotcr/lime) — Marco Tulio Ribeiro
