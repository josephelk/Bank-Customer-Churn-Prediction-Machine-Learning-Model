# Bank-Customer-Churn-Prediction-Machine-Learning-Model

🚀 Project Snapshot — Bank Customer Churn Prediction
🔎 Project Overview

Built an end-to-end machine learning pipeline to predict customer churn in a retail banking dataset. The project covers EDA, preprocessing, model benchmarking, hyperparameter tuning, and evaluation.

🏆 Highlights

Explored customer demographics, behavior, and account details to uncover churn patterns.

Engineered features by handling irrelevant columns, encoding categoricals, and scaling numerics.

Compared models (LogReg, KNN, SVM, Decision Tree, Random Forest, XGBoost) to identify the most effective approach.

Optimized Random Forest with GridSearchCV → achieving highest ROC AUC.

📂 Dataset at a Glance

Size: ~10K customers

Target: churn (1 = churned, 0 = retained)

Features: demographics (age, gender, country), account metrics (credit score, balance, products, tenure, salary), engagement (credit card, active status)

📊 Key Insights from EDA

Churn imbalance: majority are retained customers

Age & balance: higher churn tendency in middle-aged customers with mid-level balances

Active status: inactive customers churn more frequently

Number of products: strong influence on churn probability

🧠 Modeling & Evaluation

Best Model: Random Forest Classifier

Tuning: GridSearchCV (ROC AUC as primary scoring)

Performance (tuned RF):

Accuracy: ~0.86

Recall (churn class): ~0.70

Precision: ~0.85

F1: ~0.77

ROC AUC: High

🎯 Impact

Provides an interpretable model for customer retention strategies.

Identifies at-risk customers early, allowing proactive engagement.

Demonstrates a scalable ML pipeline (ColumnTransformer + Pipeline) for real-world datasets.

📚 What I Learned

Designing leak-free ML pipelines with train/test separation.

Practical use of GridSearchCV for hyperparameter tuning.

Communicating model performance using ROC curves and confusion matrices.

Balancing business needs (catch more churners) with technical metrics (precision vs recall).
