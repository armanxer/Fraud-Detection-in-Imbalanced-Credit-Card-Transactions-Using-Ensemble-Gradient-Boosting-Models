# Fraud Detection in Imbalanced Credit Card Transactions Using Ensemble & Gradient Boosting Models

### Comparative Analysis of Ensemble and Gradient Boosting Models

A machine learning project focused on detecting fraudulent credit card transactions from a highly imbalanced dataset of **284,807 transactions**, containing only **492 fraud cases (0.172%)**.

## Project Highlights

* Performed exploratory data analysis and class-imbalance assessment
* Analyzed transaction amount, time, and PCA-transformed features
* Built and compared multiple classification models:

  * Random Forest
  * AdaBoost
  * CatBoost
  * XGBoost
  * LightGBM
* Evaluated models using **ROC-AUC**, confusion matrices, and feature importance
* Applied train-validation-test splitting and cross-validation
* Identified influential features associated with fraudulent transactions
* Achieved the strongest performance with **XGBoost**, reaching approximately **0.984 validation ROC-AUC** and **0.974 test ROC-AUC**

## Tech Stack

`Python` • `Pandas` • `NumPy` • `Scikit-learn` • `XGBoost` • `LightGBM` • `CatBoost` • `Matplotlib` • `Seaborn` • `Plotly`

## Skills Demonstrated

**Machine Learning • Binary Classification • Imbalanced Data Analysis • EDA • Model Evaluation • Ensemble Learning • Gradient Boosting • Feature Importance • Cross-Validation • Data Visualization**

## Outcome

The project demonstrates an end-to-end machine learning workflow for fraud detection, with comparative model evaluation showing **XGBoost as the best-performing approach** on unseen transaction data.
