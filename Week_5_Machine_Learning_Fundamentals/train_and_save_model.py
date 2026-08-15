import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Key features only (top drivers identified across Decision Tree / RF / XGBoost in earlier tasks)
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = ['Contract', 'OnlineSecurity', 'TechSupport', 'InternetService',
                         'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES].copy()
# SeniorCitizen is 0/1 int in the raw data — treat as categorical string for consistent one-hot encoding
X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)
y = (df['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
])

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                                      random_state=42, n_jobs=-1)),
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy: ", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:   ", recall_score(y_test, pred))
print("F1:       ", f1_score(y_test, pred))
print("ROC-AUC:  ", roc_auc_score(y_test, proba))

# Save the full pipeline (preprocessing + model) as a single deployable artifact
joblib.dump(pipeline, 'churn_model.joblib')

# Save the exact category choices so the Streamlit app can build correct dropdowns
choices = {col: sorted(df[col].astype(str).unique().tolist()) if col != 'SeniorCitizen'
           else ['0', '1'] for col in CATEGORICAL_FEATURES}
joblib.dump({'numeric_features': NUMERIC_FEATURES,
             'categorical_features': CATEGORICAL_FEATURES,
             'choices': choices}, 'churn_model_meta.joblib')

print("\nSaved churn_model.joblib and churn_model_meta.joblib")
