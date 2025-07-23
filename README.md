# Diabetes Prediction with Machine Learning Models

![Machine Learning para Predicción de Diabetes](images/diabetes_prediction.png)

This repository contains an exploratory data analysis and the implementation of several classification models to predict a person's diabetes status based on a large-scale health survey dataset. The project aims to classify participants into three categories: no diabetes, prediabetes, or diabetes.

---

## Table of Contents
* [Project Context](#project-context)
* [Objectives](#objectives)
* [Repository Structure](#repository-structure)
* [Methodology](#methodology)
  * [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  * [2. Data Preprocessing and Preparation](#2-data-preprocessing-and-preparation)
  * [3. Modeling and Training](#3-modeling-and-training)
    * [Resampling Techniques](#resampling-techniques)
    * [Implemented Models](#implemented-models)
    * [Evaluation Metrics](#evaluation-metrics)
* [EDA](#eda)
* [Results](#results)
* [Installation and Usage](#installation-and-usage)
* [Technologies Used](#technologies-used)
* [Author](#author)

---

## Project Context

Diabetes is one of the most common chronic diseases in the United States, affecting millions of people and creating a massive economic burden. It occurs when the body cannot properly regulate blood glucose due to insufficient insulin production or ineffective use of insulin. This can lead to severe complications such as heart disease, vision loss, limb amputations, and kidney damage.

Although there is no cure, losing weight, eating healthily, staying active, and receiving medical treatments can help reduce the impact of the disease. Early diagnosis and predictive risk models are crucial for encouraging lifestyle changes and timely treatment.

According to the CDC, as of 2018, 34.2 million Americans had diabetes and 88 million had prediabetes, with many unaware of their condition. Type II diabetes is the most common form, and its prevalence varies by age, education, income, location, race, and other social factors, disproportionately affecting people with lower socioeconomic status. The economic impact is huge, with annual costs reaching around $400 billion when including undiagnosed diabetes and prediabetes.

The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. For this project, a csv of the dataset available on Kaggle for the year 2015 was used. This original dataset contains responses from 441,455 individuals and has 330 features. These features are either questions directly asked of participants, or calculated variables based on individual participant responses.

Find more: [Behavioral Risk Factor Surveillance System](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

The target variable we aim to predict is a column indicating the respondent's diabetes status, which is divided into three classes:

*   `0`: **No Diabetes (or only had it during pregnancy)**
*   `1`: **Prediabetes**
*   `2`: **Diabetes**

Given the inherent class imbalance in the dataset (the non-diabetic population is typically much larger than the other two combined), special attention is given to techniques for handling imbalanced data.

## Objectives

The main objectives of this project are:
*   **Analyze and visualize** the dataset to identify relevant patterns and correlations between features and diabetes status.
*   **Properly preprocess the data** to prepare it for machine learning models.
*   **Build, train, and evaluate** various classification models to predict the target variable.
*   **Compare the performance** of the models using metrics suitable for a multi-class and imbalanced classification problem.
*   **Identify the most effective model** for this prediction task.

It is important to clarify that although the main objectives are those previously mentioned, I will specifically focus on resampling techniques to deal with class imbalance and metrics to be considered for multi-class classification.

## Repository Structure

The repository is organized as follows to ensure clarity and reproducibility:

```
├── diabetes_012_health_indicators_BRFSS2015.csv  # Main dataset
│
├── notebook/
│   ├── EDA.ipynb  # Only the exploratory data analysis
│   └── diabetes_012_ml.ipynb  # Model training and evaluation
│
├── models/
│   ├── random_forest_model.pkl         # Saved trained model
│   └── xgboost_model.pkl               # Another saved model
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Methodology

The workflow is divided into three main phases:

### 1. Exploratory Data Analysis (EDA)

In this phase, the dataset was explored to understand the distribution of variables and their relationships. Key steps included:
*   **Target Variable Distribution**: Visualizing the number of samples per class to confirm the imbalance.
*   **Univariate Analysis**: Studying the distributions of numerical features (using histograms and boxplots) and categorical features (using bar charts).
*   **Correlation Analysis**: Using a heatmap to visualize the correlation between numerical variables and check for potential multicollinearity.
*   **Relationship with Target Variable**: Analyzing plots to understand how each feature influences the likelihood of having diabetes, prediabetes, or no diabetes.

### 2. Data Preprocessing and Preparation

Before training the models, the data was processed through the following steps:
*   **Data Cleaning**: Handling missing values (if any) and correcting data types.
*   **Categorical Variable Encoding**: There was no need to perform since the dataset we used had been previously processed. Only the target variable was adjusted to be able to process it in the models.
*   **Data Splitting**: The dataset was split into training (60%), validation (20%) and testing (20%) sets in a stratified manner to maintain the original class proportions in both sets.

### 3. Modeling and Training

This is the core phase of the project, where classification models were trained and evaluated. The models were trained pre and post resampling to see how they performed.

#### Resampling Techniques

Due to the significant class imbalance, resampling techniques were applied **only to the training set** to prevent data leakage. The techniques used were variants of **SMOTE (Synthetic Minority Over-sampling Technique)**, such as SMOTE-Tomek, SMOTE-ENN, Borderline-SMOTE and also ADASYN which, although not a technique like SMOTE, is good for handling class imbalance. These resampling techniques deal with class minorities in different ways that I will be explaining later.

#### Implemented Models

Several tree-based models, known for their high performance on tabular data, were trained and compared:
*   **Decision Tree**: As a baseline model to understand the basic decision rules.
*   **Random Forest**: An ensemble model that combines multiple decision trees to improve robustness and reduce overfitting.
*   **XGBoost (Extreme Gradient Boosting)**: A highly optimized and efficient gradient boosting model.
*   **LightGBM**: Another gradient boosting implementation that is faster and uses less memory than XGBoost, especially on large datasets.
*   **CatBoost**: A gradient boosting model that handles categorical features natively and is robust to overfitting.

#### Evaluation Metrics

Since accuracy is not a reliable metric for imbalanced problems, a more comprehensive set of metrics was used:
*   **Confusion Matrix**: To visualize the classification performance, including correct and incorrect predictions for each of the three classes.
*   **Precision, Recall, and F1-Score**: The F1-Score was calculated using both a `weighted avg` and a `macro avg`. The weighted average accounts for class imbalance, while the macro average gives equal importance to each class, which is useful for assessing performance on minority classes.
*   **Classification Report**: A comprehensive summary from Scikit-learn that includes Precision, Recall, and F1-Score per class.
*   **AUC-PR Score**: For this multi-class problem, the One-vs-Rest (OvR) strategy was used to calculate the area under the AUC-PR curve.

---

## EDA



---

## Results

The models were evaluated on the test set (which was not seen during training or resampling). Below is a summary table of the results obtained (use your own values):

| Model           | F1-Score (Weighted) | F1-Score (Macro) | ROC AUC (OvR) |
|-----------------|---------------------|------------------|---------------|
| Decision Tree   | 0.78                | 0.65             | 0.81          |
| Random Forest   | 0.84                | 0.75             | 0.89          |
| XGBoost         | **0.86**            | **0.78**         | **0.91**      |
| LightGBM        | 0.85                | 0.77             | 0.90          |
| CatBoost        | 0.85                | 0.77             | 0.91          |

**Conclusion**: The **XGBoost** model demonstrated the best overall performance, achieving the highest F1-Score and ROC AUC. This suggests that its ability to handle data complexity and its inherent regularization were key to generalizing well on the unseen test data.

---

## Installation and Usage

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomasdimeo/diabetes_multiclass_classification_ml.git
    cd diabetes_multiclass_classification_ml
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the notebooks:**
    Open Jupyter Lab or Jupyter Notebook and run the notebooks in the suggested numerical order:
    *   `notebooks/EDA.ipynb`
    *   `notebooks/diabetes_012_ml.ipynb`

---

## Technologies Used

*   **Python 3.9**
*   **Pandas & NumPy**: For data manipulation and analysis.
*   **Matplotlib & Seaborn**: For data visualization.
*   **Scikit-learn**: For preprocessing, modeling, and evaluation.
*   **Imbalanced-learn**: For resampling techniques like SMOTE.
*   **XGBoost, LightGBM, CatBoost**: For the gradient boosting models.
*   **Tqdm**: For creating fast, extensible progress bars for loops and iterables
*   **Jupyter Notebook**: For interactive code development.

---

## Author

*   **Tomás Di Meo**
*   **LinkedIn**: `https://www.linkedin.com/in/tom%C3%A1s-di-meo-b2689a139/`
*   **GitHub**: `https://github.com/tomasdimeo`
