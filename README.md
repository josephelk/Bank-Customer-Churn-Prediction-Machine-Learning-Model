# 📉 Bank Customer Churn Prediction

> Predicting which bank customers are likely to churn using an end-to-end machine learning pipeline (EDA → preprocessing → modeling → tuning → evaluation).

---

## 🔎 Project Overview
Built a **binary classification model** to identify customers at risk of leaving the bank.  
The project covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing (encoding, scaling, feature selection)
- Model benchmarking & selection
- Hyperparameter tuning with GridSearchCV
- Model evaluation & interpretation

---

## 🏆 Highlights
- 📊 **EDA** revealed churn imbalance, age & activity as strong churn indicators  
- ⚙️ **Pipeline-based preprocessing** to prevent data leakage  
- 🤖 **Model comparison**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, XGBoost  
- 🌟 **Random Forest (tuned)** achieved the best ROC AUC score  

---

## 🧾 Dataset
| Column            | Description                              |
|-------------------|------------------------------------------|
| `customer_id`     | Unique customer identifier (dropped)     |
| `credit_score`    | Creditworthiness score                   |
| `country`         | Customer’s country of residence          |
| `gender`          | Male/Female                              |
| `age`             | Customer’s age                           |
| `tenure`          | Years stayed with bank                   |
| `balance`         | Account balance                          |
| `num_of_products` | Number of products used                  |
| `credit_card`     | Owns a credit card (1 = Yes, 0 = No)     |
| `active_member`   | Active status (1 = Yes, 0 = No)          |
| `estimated_salary`| Estimated annual salary                  |
| `churn`           | Target (1 = churned, 0 = retained)       |

---

## 📊 EDA Insights
- Majority customers are retained → dataset is **imbalanced**  
- Middle-aged customers with medium balances are more prone to churn  
- Inactive customers show **significantly higher churn rate**  
- Number of products strongly correlates with churn  

---

## 🧠 Modeling
- Models tested:  
  `Logistic Regression | KNN | SVM | Decision Tree | Random Forest | XGBoost`  
- **Random Forest** selected as final model (best trade-off of recall & precision)  

### ⚙️ Fine-Tuning (GridSearchCV)
```python
param_grid = {
  "n_estimators": [200, 400, 800],
  "max_depth": [None, 10, 20, 30],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4],
  "max_features": ["sqrt", "log2"]
}
```

## 📈 Model Performance (Tuned Random Forest)

| Metric           | Score  |
|------------------|--------|
| Accuracy         | ~0.86  |
| Recall (Churn)   | ~0.70  |
| Precision        | ~0.85  |
| F1-Score         | ~0.77  |
| ROC AUC          | High   |

---

## 🎯 Impact
- Enables banks to **identify at-risk customers** early  
- Helps target **retention campaigns** more efficiently  
- Provides an **interpretable ML workflow** (feature importance, ROC analysis)  

---

## 📚 What I Learned
- How to build **end-to-end ML pipelines** with scikit-learn  
- Importance of avoiding **data leakage** (fit only on train data)  
- Using **GridSearchCV** for model optimization  
- Interpreting results with **ROC curve & confusion matrix**  
- Balancing business needs (catch churners) vs technical metrics  

---

## 📌 Conclusion
- **Random Forest** delivered the most reliable results among all models tested.  
- The **ROC Curve & AUC score** were crucial in evaluating model quality beyond accuracy.  
- Key churn drivers: `credit_score`, `age`, `balance`, `num_of_products`, and `active_member`.  
- The project demonstrates how machine learning can support **data-driven decision making** in banking, especially for **customer retention strategies**.  

