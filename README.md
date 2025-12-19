# Hg-MC-Auto: An End-to-End Self-Driving Pipeline for Mercury Isotope Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Yoko286/Hg-MC-Auto)](https://github.com/Yoko286/Hg-MC-Auto/stargazers)

**Hg-MC-Auto** is a comprehensive, intelligent pipeline for automated mercury isotope analysis by MC-ICP-MS, integrating robotic data extraction, expert-informed quality control, and machine learning diagnostics.

![Graphic Abstract](https://github.com/Yoko286/Hg-MC-Auto/blob/main/docs/Graphic%20abstract.png)

## ✨ Features

- **Automated Data Processing**: Robotic export from proprietary software to structured formats
- **Intelligent Quality Control**: Hierarchical ML models with 99.6% F1-score accuracy
- **Root Cause Diagnosis**: Multi-class classification for anomaly identification
- **User-Friendly Interface**: Interactive GUI with configurable options
- **Scalable Framework**: Modular design for extension to other isotope systems
- **Expert-Validated**: Built on 27,000+ historical MC-ICP-MS measurements

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Usage Guide](#usage-guide)
6. [Models](#models)
7. [Citation](#citation)
8. [Contact](#contact)

## 🚀 Installation

**Prerequisites**
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for large datasets)
- MC-ICP-MS raw data files (.dat and .log formats)

**Step-by-Step Setup**


### 1. Clone the repository
```bash
git clone https://github.com/Yoko286/Hg-MC-Auto.git
```
### 2. Navigate to the project directory
```bash
cd Hg-MC-Auto
```
### 3. Create and activate a virtual environment
```bash
python -m venv hg-auto-env
```
### On Windows:
```bash
hg-auto-env\Scripts\activate
```
### On macOS/Linux:
```bash
source hg-auto-env/bin/activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Run the application
```bash
python src/main.py
```
🏗️ Architecture
Hg-MC-Auto employs a three-tier architecture that transforms raw MC-ICP-MS data into quality-assured isotopic results:

![Overall Architecture Diagram](https://github.com/Yoko286/Hg-MC-Auto/blob/main/docs/Overall%20Architecture%20Diagram.png)

Core Components
Data Automation Layer

Robotic Process Automation (RPA) for data extraction

Automated file conversion and merging

Batch processing of historical data

Intelligence Layer

Expert rule-based filtering

Ensemble machine learning models

Hierarchical anomaly detection

Application Layer

Interactive graphical interface

Configurable quality thresholds

Comprehensive reporting

## 📁 Project Structure
```bash
Hg-MC-Auto/
│
├── data/                     # Sample datasets for testing
│   └── tangyang.xlsx        # Model validation dataset
│
├── docs/                     # Documentation and figures
│
├── model/                    # Pre-trained ML models
│   ├── Exter_ML_model/      # Binary classification models
│   │   ├── top3_model_1_Random_Forest_SMOTE.pkl
│   │   ├── top3_model_2_Bagging_RF_UnderSampling.pkl
│   │   └── top3_model_3_XGBoost_UnderSampling.pkl
│   │
│   └── Inter_ML_model/      # Multi-class diagnostic models
│       ├── Basic_features/  # Core feature-based model
│       └── Enhanced_features/ # Advanced feature-based model
│
├── results/                  # Output directory for analysis results
│
├── src/                      # Source code
│   ├── main.py              # Main application entry point
│   ├── 1_Automatic_export.py          # Data export automation
│   ├── 2_Automatic_calculation.py     # Isotope ratio calculation
│   ├── 3_Empirical_model.py           # Expert rule-based classification
│   ├── 4_ML_Predict.py                # ML model prediction interface
│   ├── 5_Exter_ML_train.py            # Binary classifier training
│   └── 6_Inter_ML_train.py            # Multi-class classifier training
│
├── custom_ranges_config.json          # User-configurable acceptance ranges
├── mouse_coordinates.config           # RPA coordinate settings
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```
## 📖 Usage Guide
Launching the Application
After installation, run:

```bash
python src/main.py
```

You will see the interactive interface:

```bash
Welcome to Hg_MC_Auto!
============================================================

Please select a task:
1. Automatically export isotope data
2. Automatically export instrument parameters, merge isotope data, and calculate isotope fractionation values
3. Classify data using an empirical model
4. Classify data using a machine learning model
5. Train your own machine learning expert model
0. Exit
```
## Task Options Explained
Option 1: Automated Data Export
Converts proprietary .dat files to structured CSV format

Merges with corresponding instrument log files

Uses RPA for vendor software interaction

Option 2: Isotope Calculation
Calculates δ202Hg values relative to NIST SRM 3133

Computes mass-independent fractionation anomalies (Δ-values)

Batch processes entire datasets

Option 3: Empirical Model Classification
Applies literature-based acceptance ranges (Table 1 in manuscript)

Flags measurements outside 95% confidence intervals

User-configurable thresholds via custom_ranges_config.json

Option 4: ML Model Prediction
Uses pre-trained ensemble models for quality assessment

Provides confidence scores for each prediction

Identifies probable causes for abnormal measurements

Option 5: Custom Model Training
Train laboratory-specific models using your annotated data

Supports both binary and multi-class classification

Adapts to different instrument performances and sample matrices

## 🤖 Models
Binary Classification Models
Purpose: Distinguish between "Normal" and "Abnormal" measurements

Performance: Test F1-score: 0.9960, AUC: 0.999-1.0

Algorithms: Random Forest, XGBoost, Bagging Classifiers

Sampling Strategies: SMOTE, ADASYN, SMOTEENN, UnderSampling

Multi-class Diagnostic Models
Purpose: Identify root causes of abnormalities

Categories:

"Possible instrument instability"

"Potential concentration anomaly"

"Combined factors"

"Other reasons, retesting recommended"

Features: Internal precision metrics, concentration mismatch ratios

## 📊 Performance Highlights
Metric	Binary Classification	Multi-class Diagnosis
Accuracy	99.61%	99.84%
F1-Score	0.9960	0.9909 (balanced)
Recall (Normal)	99.8%	-
AUC	0.999-1.0	-
Based on validation with 27,942 historical measurements

## 📝 Citation
If you use Hg-MC-Auto in your research, please cite:

bibtex
@article{zhou2025selfdriving,
  title={A Self-Driving Pipeline for Mercury Isotope Analysis: From Automated Data Acquisition to AI-Powered Quality Diagnosis},
  author={Zhou, Chufan and Huang, Qiang and Tang, Yang and Zhong, Ying and Feng, Xinbin},
  journal={AC},
  year={2025},
  doi={10.xxxx/xxxxxx}
}
## 🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch

Submit a pull request

Ensure code follows PEP 8 guidelines

Include tests for new functionality

## 🐛 Issues and Support
Bug Reports: Use the GitHub Issues page

Questions: Check the Wiki or open a discussion

Feature Requests: Submit via GitHub Issues with the "enhancement" label

## 📧 Contact
State Key Laboratory of Environmental Geochemistry
Institute of Geochemistry, Chinese Academy of Sciences
Guiyang, Guizhou 550081, China

Chufan Zhou:
📧 zhouchufan@mail.gyig.ac.cn
🔗 ORCID: [0009-0008-0144-9017](https://orcid.org/0009-0008-0144-9017)

Qiang Huang (Corresponding Author):
📧 huangqiang@mail.gyig.ac.cn
🔗 ORCID: [0000-0003-1568-9042](https://orcid.org/0000-0003-1568-9042)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.