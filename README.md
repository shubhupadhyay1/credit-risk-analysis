

# **Credit Risk Analysis: Predicting Defaults with Machine Learning**

---

### **Table of Contents**

1. [Introduction](#introduction)  
2. [Motivation](#motivation)  
3. [Methodology](#methodology)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Feature Engineering](#feature-engineering)  
   - [Data Scaling & Handling Imbalance](#data-scaling--handling-imbalance)  
   - [Model Training & Custom Evaluation](#model-training--custom-evaluation)  
   - [Model Comparison](#model-comparison)  
   - [Generating Predictions](#generating-predictions)  
4. [Key Insights](#key-insights)  
5. [Conclusion](#conclusion)  
6. [Future Work](#future-work)

---

## **1. Introduction**

In financial lending, **risk is everything**. Every borrower represents a probability—will they repay the loan or default? Poorly predicted defaults can cause massive losses, destabilizing entire financial institutions. Accurate **credit risk analysis** isn’t just a statistical problem; it’s a survival strategy.

Unlike generic machine learning pipelines, this project was built with deep consideration of **finance-specific metrics** like **Information Value (IV)** and **Weight of Evidence (WOE)**, ensuring the models aren’t just accurate but **interpretable and actionable**.

---

## **2. Motivation**

Traditional credit risk models often rely on static statistical methods that fail to capture complex, non-linear relationships in data. We wanted to push beyond these limitations by building:

1. **A robust pipeline** that handles data preprocessing, feature selection, scaling, and class imbalance effectively.
2. **Multiple machine learning models** with a custom evaluation framework.
3. **A solution with high interpretability**, making it practical for real-world financial institutions to adopt.

---

## **3. Methodology**

Our workflow is broken down into several key stages:

### **3.1 Data Preprocessing**

#### **Steps Taken**:
1. **Imputation of Missing Values**:
   - Categorical features were imputed using their **mode**.
   - Numerical features were imputed using the **median** to reduce the impact of outliers.

2. **Dropping Unnecessary Columns**:
   Features like `customer_id` and `name` were dropped as they do not contribute to the prediction task.

---

### **3.2 Feature Engineering**

Feature engineering was a crucial step in this project, involving both **statistical filtering** and **transformations tailored to financial data**.

#### **Information Value (IV) Filtering**:
- We computed the **Information Value (IV)** for each feature to assess its predictive power.
- Features with **IV < 0.02** were dropped, ensuring that only the most relevant features were retained.
  
  _IV quantifies the strength of a feature’s relationship with the target variable—higher IV means stronger predictive power._

#### **Weight of Evidence (WOE) Transformation**:
- After IV filtering, we applied **WOE binning** to all remaining features.
- WOE scales features in a way that ensures a **monotonic relationship with the target**, which is crucial for models like Logistic Regression.

---

### **3.3 Data Scaling & Handling Imbalance**

- **Scaling**:  
  We applied **Min-Max scaling** to normalize feature values, ensuring compatibility across different models.
  
- **Class Imbalance**:  
  Since the dataset had significantly fewer defaults than non-defaults, we used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the classes. This ensures that models don’t become biased toward predicting non-defaults.

---

### **3.4 Model Training & Custom Evaluation**

#### **Models Trained**:
We trained the following models:

1. **Logistic Regression**: A baseline model, valued for its simplicity and interpretability.
2. **Decision Tree**: Offers inherent interpretability but prone to overfitting.
3. **Random Forest**: An ensemble of decision trees that reduces overfitting.
4. **XGBoost**: A gradient-boosting model known for its high performance on tabular data.
5. **CatBoost**: Another gradient-boosting model, particularly effective for categorical data.
6. **LightGBM**: A highly efficient gradient-boosting model.
7. **K-Nearest Neighbors (KNN)**: Included for comparison, with `k=5` chosen based on error analysis.

#### **Custom Evaluation Function**:
We created a **custom evaluation function** to compute and display key metrics:

- **Accuracy**: Overall correctness of predictions.
- **F1-Score**: Balances precision and recall, crucial for imbalanced datasets.
- **AUC-ROC**: Measures a model’s ability to distinguish between defaulters and non-defaulters.

---

### **3.5 Model Comparison**

Here’s the final comparison of all models:

| Model                | Train Accuracy | Test Accuracy | Train F1 Score | Test F1 Score | AUC-ROC |
|----------------------|----------------|---------------|----------------|---------------|---------|
| Decision Tree        | 95.40%         | 95.63%        | 95.47%         | 95.70%        | 98.98%  |
| **CatBoost**         | **95.37%**     | **95.63%**    | **95.44%**     | **95.70%**    | **99.05%**  |
| Random Forest        | 95.40%         | 95.63%        | 95.47%         | 95.69%        | 98.99%  |
| LightGBM             | 95.37%         | 95.63%        | 95.42%         | 95.68%        | 99.04%  |
| XGBoost              | 95.29%         | 95.48%        | 95.36%         | 95.55%        | 99.03%  |
| KNN                  | 95.03%         | 95.16%        | 95.08%         | 95.20%        | 98.27%  |
| Logistic Regression  | 94.30%         | 94.44%        | 94.40%         | 94.52%        | 98.78%  |

**CatBoost emerged as the best-performing model**, with the highest **AUC-ROC (99.05%)** and near-perfect accuracy and F1-score.

---

### **3.6 Generating Predictions**

After selecting **CatBoost** as the best model, we trained it on the entire balanced train dataset and generated predictions on the test dataset. The predictions were saved in the file:

```
/reports/test_predictions_catboost.csv
```

---

## **4. Key Insights**

1. **CatBoost outperformed all other models**, making it the ideal choice for deployment in real-world scenarios.
2. **IV filtering and WOE binning significantly improved model interpretability**, which is crucial for financial decision-making.
3. **SMOTE balanced the dataset effectively**, ensuring that the models didn’t become biased toward predicting non-defaults.

---

## **5. Conclusion**

This project reimagines credit risk analysis by integrating advanced machine learning techniques with carefully crafted, finance-specific feature engineering. We present a solution that doesn’t just predict credit defaults with high accuracy but does so in a way that’s both insightful and actionable for real-world financial decision-making.

---

## **6. Future Work**

1. **Hyperparameter Tuning**:  
   Fine-tune the hyperparameters of the best-performing models to squeeze out even better performance.
   
2. **Explainability Tools**:  
   Integrate tools like **SHAP** or **LIME** to provide detailed explanations of individual predictions.

3. **Deployment**:  
   Deploy the final model as a **Flask API** or **Streamlit app** for real-time credit risk assessment.

---
## **Creator** 👨‍💻

If you’re curious about the project or want to collaborate, feel free to connect:

[![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/shubhupadhyay1)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shubh-upadhyay/)  
[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/shubh_upadhyayy)

---