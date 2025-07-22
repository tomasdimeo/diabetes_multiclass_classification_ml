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
* [Results](#results)
* [Installation and Usage](#installation-and-usage)
* [Technologies Used](#technologies-used)
* [Author](#author)

---

## Project Context

The dataset used in this project is derived from a large-scale health survey, likely a version of the **Behavioral Risk Factor Surveillance System (BRFSS)**. It contains over 200,000 records and 22 features for each participant. These features include health indicators (e.g., high cholesterol, high blood pressure), lifestyle habits (e.g., smoking, physical activity), and demographic data (e.g., age, sex).

The target variable we aim to predict is a column indicating the respondent's diabetes status, which is divided into three classes:

*   `0`: **No Diabetes**
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

## Repository Structure

The repository is organized as follows to ensure clarity and reproducibility:

```
├── data/
│   └── diabetes_health_indicators.csv  # Main dataset
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb  # Exploratory data analysis and data cleaning
│   └── 02_Modeling_and_Evaluation.ipynb  # Model training and evaluation
│
├── src/
│   ├── utils.py                        # Helper functions (e.g., for plotting or evaluation)
│   └── model_pipeline.py               # Scripts for the training pipeline (optional)
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
*   **Categorical Variable Encoding**: Categorical features were converted into a numerical format using techniques like One-Hot Encoding.
*   **Feature Scaling**: Numerical features were scaled (e.g., using `StandardScaler`) to ensure that models sensitive to feature scales would perform correctly and to aid the convergence of others.
*   **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets in a stratified manner to maintain the original class proportions in both sets.

### 3. Modeling and Training

This is the core phase of the project, where classification models were trained and evaluated.

#### Resampling Techniques

Due to the significant class imbalance, resampling techniques were applied **only to the training set** to prevent data leakage. The primary technique used was **SMOTE (Synthetic Minority Over-sampling Technique)**, which creates synthetic samples of the minority classes (prediabetes and diabetes) to balance the training dataset.

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
*   **ROC AUC Score**: For this multi-class problem, the One-vs-Rest (OvR) strategy was used to calculate the area under the ROC curve.

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
    git clone https://github.com/[your-username]/[your-repository].git
    cd [your-repository]
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
    *   `notebooks/01_EDA_and_Preprocessing.ipynb`
    *   `notebooks/02_Modeling_and_Evaluation.ipynb`

---

## Technologies Used

*   **Python 3.9**
*   **Pandas & NumPy**: For data manipulation and analysis.
*   **Matplotlib & Seaborn**: For data visualization.
*   **Scikit-learn**: For preprocessing, modeling, and evaluation.
*   **Imbalanced-learn**: For resampling techniques like SMOTE.
*   **XGBoost, LightGBM, CatBoost**: For the gradient boosting models.
*   **Jupyter Notebook**: For interactive code development.

---

## Author

*   **[Your Name]**
*   **LinkedIn**: `https://www.linkedin.com/in/[your-username]`
*   **GitHub**: `https://github.com/[your-username]`
