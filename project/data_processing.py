import pandas as pd

# Binary mappings used consistently in training and inference
binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop customerID
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric and fill NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Process SeniorCitizen as int
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

    # Encode binary columns using predefined mapping
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # Feature engineering
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0,12,24,36,48,60,72],
                                labels=['0-12','12-24','24-36','36-48','48-60','60-72'])

    df['MonthlyCharges_Tenure'] = df['MonthlyCharges'] * df['tenure']

    df['Contract_PaymentMethod'] = df['Contract'].astype(str) + "_" + df['PaymentMethod'].astype(str)

    # Encode other categorical columns using factorize (ensures numerical encoding)
    cat_cols = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract_PaymentMethod', 'tenure_group']

    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col], _ = pd.factorize(df[col])

    return df

def split_X_y(df, target='Churn'):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y
