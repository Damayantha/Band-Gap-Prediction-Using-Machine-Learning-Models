# Band Gap Prediction Using Machine Learning Models

## Overview

This Colab notebook implements a comprehensive machine learning pipeline to predict the band gap of materials using features extracted from their compositions and structures via the Matminer library. The goal is to accelerate the discovery of new materials by accurately estimating electronic properties with various regression models.

The dataset includes data from the Materials Project API, with features engineered using Matminer featurizers. Several state-of-the-art models were trained and evaluated to compare their performance.
[Colab](https://colab.research.google.com/drive/1Qh_cF-rDN-k-jezdYJ8OeYdzNKAqgFIr?usp=sharing) 
---

## Methodology

1. **Data Acquisition & Preprocessing**  
   - Materials data fetched via Materials Project API  
   - Features extracted using Matminer featurizers for compositions and structures  
   - Dataset cleaned and NaN features removed

2. **Feature Engineering**  
   - Use of multiple featurizers to capture chemical and structural information  
   - Feature concatenation and normalization

3. **Model Training & Evaluation**  
   - Models trained:  
     - XGBoost Regressor  
     - Gradient Boosting Regressor  
     - Random Forest Regressor  
     - Stacked Trees (Ensemble)  
     - Multi-layer Perceptron (MLP) Regressor  
     - Support Vector Regressor (SVR)  
     - Linear Regression  
     - Ridge Regression  
     - Decision Tree Regressor  
     - AdaBoost Regressor  
     - K-Nearest Neighbors (KNN)  
     - ElasticNet  
   - Performance evaluated using:  
     - Coefficient of Determination (R²)  
     - Root Mean Squared Error (RMSE)

---

## Results

### Initial Model Performance

| Model              | R²      | RMSE     |
|--------------------|---------|----------|
| XGBoost            | 0.8600  | 0.5528   |
| Gradient Boosting  | 0.8588  | 0.5553   |
| Random Forest      | 0.8452  | 0.5814   |
| Stacked Trees      | 0.8327  | 0.6045   |
| MLP Regressor      | 0.7847  | 0.6857   |
| SVR                | 0.7677  | 0.7123   |
| Linear Regression  | 0.6958  | 0.8150   |
| Ridge Regression   | 0.6914  | 0.8209   |
| Decision Tree      | 0.6563  | 0.8663   |
| AdaBoost           | 0.6050  | 0.9288   |
| KNN                | 0.3726  | 1.1705   |
| ElasticNet         | 0.1324  | 1.3764   |

---

### Optimized Model Performance

| Model              | R²      | RMSE     |
|--------------------|---------|----------|
| Gradient Boosting  | 0.8827  | 0.5178   |
| XGBoost            | 0.8811  | 0.5213   |
| Random Forest      | 0.8620  | 0.5617   |
| Stacked Trees      | 0.8570  | 0.5718   |
| MLP Regressor      | 0.8446  | 0.5960   |
| SVR                | 0.8010  | 0.6745   |
| Decision Tree      | 0.7250  | 0.7930   |
| Linear Regression  | 0.7237  | 0.7948   |
| Ridge Regression   | 0.7235  | 0.7951   |
| AdaBoost           | 0.6758  | 0.8609   |
| KNN                | 0.3734  | 1.1969   |
| ElasticNet         | 0.1679  | 1.3793   |

---

## Usage

1. Clone or open the Colab notebook linked in this repository.
2. Make sure you have an API key for Materials Project (MP API) and set it up in the notebook.
3. Run all cells sequentially to:
   - Download and preprocess the dataset
   - Extract features using Matminer featurizers
   - Train and evaluate machine learning models
4. Modify hyperparameters or featurizers as needed to improve performance.

---

## Dependencies

- Python 3.x  
- [Matminer](https://hackingmaterials.github.io/matminer/)  
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)  
- scikit-learn  
- pandas  
- numpy  

Install all dependencies using:

```bash
pip install matminer xgboost scikit-learn pandas numpy
