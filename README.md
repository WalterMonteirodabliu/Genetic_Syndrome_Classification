# Apollo Solutions Machine Learning Developer Test

## Project Overview

This project implements a complete machine learning pipeline to classify genetic syndromes using 320-dimensional image embeddings provided in a hierarchical pickle file.

The pipeline includes:

- Data loading and preprocessing

- Hierarchical data flattening

- Exploratory data analysis

- Dimensionality reduction (t-SNE)

- K-Nearest Neighbors classification

- Comparison between Euclidean and Cosine distance metrics

- 10-fold cross-validation

- Manual implementation of evaluation metrics (F1-score, AUC, Top-k accuracy)

- ROC curve generation

- Automatic generation of performance tables

All scripts are implemented as standalone Python modules

## Project Structure

``` apollo_ml_test/
│
├── data/
│   └── mini_gm_public_v0.1.p
│
├── src/
│   ├── data_processing.py
│   ├── visualization.py
│   ├── metrics.py
│   ├── cross_validation.py
│   └── main.py
│
├── outputs/
│
├── requirements.txt
└── README.md
```
## Setup Instructions
### 1. Create Virtual Environment

``` 
python -m venv .venv 
```
Activate:

Windows:
``` 
.venv\Scripts\activate
```
Linux / macOS:
``` 
source .venv/bin/activate
```
### 2. Install Dependencies

``` 
pip install -r requirements.txt
```

If a NumPy-related pickle loading error occurs (e.g., `numpy.core._multiarray_umath`), upgrade NumPy:

``` 
pip install --upgrade numpy
```

## Running the Project

From the project root:

``` 
python -m src.main
```

This will execute:

- Data loading and flattening

- Exploratory Data Analysis

- t-SNE dimensionality reduction

- 10-fold cross-validation for KNN

- Performance comparison between Euclidean and Cosine metrics

- ROC curve generation

## Outputs

All results are automatically saved in the `outputs/` directory:

- `eda_summary.csv`

- `images_per_syndrome.csv`

- `tsne_plot.png`

- `knn_cv_results_euclidean.csv`

- `knn_cv_results_cosine.csv`

- `knn_best_summary.csv`

- `roc_knn_microavg.png`

## Requirements

- Python 

- numpy

- pandas

- scikit-learn

- matplotlib