# File path: credit_card_fraud_detection_advanced.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('creditcard.csv')
print(data.head())
print(data.info())
print(data.describe())
print(f"Number of fraudulent transactions: {data['Class'].sum()}")

data = data.dropna()
X = data.drop('Class', axis=1)
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

rf_model = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc', n_iter=50, random_state=42)
xgb_random_search.fit(X_train, y_train)
best_xgb_model = xgb_random_search.best_estimator_

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
lr_grid_search = GridSearchCV(estimator=lr_model, param_grid=lr_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
lr_grid_search.fit(X_train, y_train)
best_lr_model = lr_grid_search.best_estimator_

models = {
    'Random Forest': best_rf_model,
    'XGBoost': best_xgb_model,
    'Logistic Regression': best_lr_model
}

for model_name, model in models.items():
    print(f"Evaluating {model_name}")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = %0.2f)' % roc_auc_score(y_test, y_prob))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

best_model = best_rf_model
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_prob
})
results.to_csv('fraud_detection_results.csv', index=False)
