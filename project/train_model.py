import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from data_processing import load_data, clean_data, split_X_y

def train_and_save_model(data_path, model_path):
    """Training LightGBM model with SMOTE and saving it"""

    # Load and clean data
    df = load_data(data_path)
    df = clean_data(df)

    # Split features and target
    X, y = split_X_y(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Apply SMOTE oversampling
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train LightGBM model
    model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("LightGBM ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model(
        "D:/projects/customer_churn_prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "models/lgb_model.pkl"
    )
