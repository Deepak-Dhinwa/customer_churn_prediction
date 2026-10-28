import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "churn_stack_pipeline.pkl"


def load_model(model_path=None):
    path = Path(model_path) if model_path else DEFAULT_MODEL
    return joblib.load(path)


def engineer_features(df):
    """Build the columns the saved stacking pipeline was fit on.

    Categorical values stay as text. The pipeline's one-hot encoder
    owns that mapping, so a new batch gets the same codes as training.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())
    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"],
    ).astype(str)
    out["MonthlyCharges_Tenure"] = out["MonthlyCharges"] * out["tenure"]
    out["Contract_PaymentMethod"] = (
        out["Contract"].astype(str) + "_" + out["PaymentMethod"].astype(str)
    )
    return out


def predict_churn(model, input_df):
    frame = engineer_features(input_df)
    missing = [col for col in model.feature_names_in_ if col not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = frame[list(model.feature_names_in_)]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result_df = input_df.copy()
    result_df["Churn_Pred"] = preds
    result_df["Churn_Prob"] = probs
    return result_df


if __name__ == "__main__":
    model = load_model()
    sample = pd.read_csv(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv", nrows=5)
    print(predict_churn(model, sample)[["Churn_Pred", "Churn_Prob"]])
