"""Logistic Regression on the PaySim dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              average_precision_score, roc_auc_score,
                              confusion_matrix)

ps = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')
# nameOrig/nameDest are unique IDs, not predictive features.
# isFlaggedFraud is a near-unused naive rule (16 of 6.36M rows) dropped to avoid leakage.
ps = ps.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

fraud = ps[ps.isFraud == 1]
nonfraud = ps[ps.isFraud == 0].sample(n=100000, random_state=42)
ps_sampled = pd.concat([fraud, nonfraud]).sample(frac=1, random_state=42).reset_index(drop=True)

X = ps_sampled.drop(columns=['isFraud'])
y = ps_sampled['isFraud']

categorical_cols = ['type']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("=== Logistic Regression on PaySim ===")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_proba):.4f}")
print(confusion_matrix(y_test, y_pred))
