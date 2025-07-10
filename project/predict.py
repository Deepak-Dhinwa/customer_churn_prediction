import pandas as pd
import joblib
from data_processing import clean_data

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_churn(model, input_df):
    """
    Predict churn for input dataframe.
    Assumes input_df is pre-loaded DataFrame of new customer data.
    """
    # Clean and preprocess input data (same as training)
    df_cleaned = clean_data(input_df)

    # Drop target if exists in input
    if 'Churn' in df_cleaned.columns:
        df_cleaned.drop('Churn', axis=1, inplace=True)

    # Predict
    preds = model.predict(df_cleaned)
    probs = model.predict_proba(df_cleaned)[:,1]

    # Return predictions and probabilities
    result_df = input_df.copy()
    result_df['Churn_Pred'] = preds
    result_df['Churn_Prob'] = probs

    return result_df

if __name__ == "__main__":
    # Example usage
    model = load_model("models/lgb_model.pkl")

    # Example new data
    new_data = pd.read_csv("data/new_customers.csv")
    result = predict_churn(model, new_data)
    print(result.head())
