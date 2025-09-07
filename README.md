# Bank-Customer-Churn-Prediction-Machine-Learning-Model

Bank Customer Churn Prediction — End-to-End ML Pipeline

Goal. Build a binary classifier that predicts whether a bank customer will churn (leave the bank). The project walks through EDA → preprocessing → model training → evaluation → interpretation, plus hyperparameter tuning with GridSearchCV.

Dataset. Bank Customer Churn Prediction.csv

Target: churn (0 = stay, 1 = churn)

Examples of features: demographics and account signals (e.g., country, gender, age, tenure, balance, num_of_products, credit_card, active_member, estimated_salary, credit_score, …).

Categorical columns: at least country, gender.

ID columns (e.g., customer_id) are dropped.

1) EDA (Exploratory Data Analysis)

Checked dataset shape, dtypes, and missing values (none reported in the notebook).

Reviewed numeric summaries (mean, std, min/max) and categorical distributions to spot imbalance and data quality.

Computed a correlation matrix for numeric features to see which variables move together (e.g., how balance, tenure, credit_score relate to churn).

Looked at class balance of churn to anticipate metric choice (precision/recall/ROC-AUC) beyond accuracy.

2) Preprocessing

Dropped identifier column (customer_id).

Encoded categoricals:

country → One-Hot Encoding (no drop; handled with OneHotEncoder).

gender → label/one-hot depending on the model.

Train/test split is performed before fitting encoders/scalers (to avoid leakage).

(If needed) Scaled numeric features with StandardScaler inside a ColumnTransformer + Pipeline.

3) Modeling

Baselines tried (imports are present in the notebook):
LogisticRegression, KNeighborsClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier.

Selected model in the notebook: Random Forest
Rationale recorded in the notebook:

Slightly higher accuracy than alternatives.

Better precision (fewer false positives).

More balanced macro/weighted averages on the classification report.

Robust and easy to implement/interpret (feature importances).

4) Evaluation

Held-out test set evaluation with accuracy and classification report (precision, recall, F1 by class).

(Recommended) Add ROC curve and ROC-AUC to assess ranking performance under class imbalance.

ROC in one line. The ROC curve plots TPR vs. FPR across thresholds; ROC-AUC summarizes this ranking quality (1.0 is perfect, 0.5 is chance). It tells you how well the model separates churners from non-churners regardless of a fixed probability cutoff.

5) Hyperparameter Tuning (GridSearchCV)

best_params_ → the hyperparameters that maximized CV ROC-AUC.

Compare test ROC-AUC, F1 (class 1), precision/recall (class 1) and the confusion matrix before vs. after tuning.

Prefer configs that raise recall for class 1 (churners) without tanking precision too much (business-dependent).

6) Why you mustn’t fit scalers/encoders on the whole dataset

Doing fit on the entire dataset (train + test) leaks information from the test set into preprocessing (means, stds, category levels), inflates CV/test scores, and breaks the fundamental principle that test data must simulate truly unseen data. Always fit on training data only, then transform validation/test data with those fitted objects.
