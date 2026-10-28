# Customer Churn Prediction

One-page summary: open `poster.html` in a browser.
Start: `.\run.ps1`

## Problem Statement (in my words)

Telecom companies constantly lose customers – this silent churn bleeds revenue. The goal here was to **understand why customers churn** and build a model to predict churn probability, so the company can act **before losing valuable users**.

---

## Approach

### Data Cleaning & Preprocessing

- Converted `TotalCharges` to numeric; filled missing values with median.
- Dropped non-informative columns like `customerID`.
- Label encoded binary columns: gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn.
- **Feature Engineering:**
  - Created **tenure groups** (0-12, 12-24, … months) to capture loyalty stages.
  - Calculated **MonthlyCharges × Tenure** as total customer value.
  - Created **Contract + PaymentMethod** interaction to model churn risk tied to payment behaviour and commitment level.

---

### Exploratory Data Analysis (EDA) and Interpretations

#### Overall Churn Rate

- **~26% churn rate**, meaning roughly **1 in 4 customers leave**. This is significant for telecom businesses as acquiring a new customer typically costs more than retaining an existing one.

#### Tenure vs Churn

- Customers in their **first year (0-12 months)** churn the most.  
  ➔ **Interpretation:** New users are at highest risk if onboarding, service quality, or expectations are mismanaged early.

#### Contract Type

- **Month-to-month contracts** have far higher churn than one or two-year contracts.  
  ➔ **Interpretation:** Longer contracts act as natural retention mechanisms by locking in users.

#### Payment Method

- Customers paying via **Electronic Check** churn the most, compared to those using credit cards or bank transfers.  
  ➔ **Interpretation:** Electronic check users may be more price sensitive or have less friction cancelling their plans.

#### Monthly Charges

- Customers with **higher monthly charges** show higher churn rates.  
  ➔ **Interpretation:** Premium plan users might leave if they feel they’re not getting equivalent value, or if competitors offer better pricing.

---

### Modelling & Training

- **Models trained:** Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Class imbalance:** Handled with **SMOTE oversampling**.
- **Ensemble:** Built a **Stacking Classifier** combining Logistic Regression, Random Forest, and XGBoost for robust performance.

---

### Evaluation Results

| Model               | ROC-AUC |
|---------------------|---------|
| Logistic Regression | ~0.80   |
| Random Forest       | ~0.83   |
| XGBoost             | ~0.83   |
| **Stacking Ensemble** | **~0.84** |

➔ **Interpretation:**  
- Ensemble learning improved model performance marginally by combining strengths of individual models.  
- ROC-AUC of **0.84** indicates the model can distinguish churned vs non-churned customers with good reliability, though further improvements could come from additional behavioural or usage data features.

---

### Model Interpretability

- **SHAP analysis** showed top drivers of churn include:
  - **Tenure length** (shorter tenure = higher churn)
  - **Contract type** (month-to-month risk)
  - **Monthly charges**
  - **Electronic check payment**

➔ **Interpretation:**  
These features are **actionable**. For example, providing onboarding offers to new customers, promoting auto-payment methods, or creating tailored bundles for high-charge users can reduce churn rates.

---

## Key Learnings

- **EDA matters:** Without understanding patterns (e.g. early tenure churn), models alone can’t drive business decisions.
- **Feature engineering** (tenure groups, payment-contract interactions) adds predictive power beyond raw data.
- **Handling class imbalance** with SMOTE improved recall on churned customers without sacrificing overall precision.
- **Interpretability tools (SHAP)** bridge the gap between technical models and business strategies, enabling trust and targeted interventions.

---

## Project Structure

```
customer_churn_prediction/
├── customer_churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_stack_pipeline.pkl          # saved stacking pipeline used by the API
├── project/
│   ├── data_processing.py
│   ├── predict.py
│   ├── train_model.py
│   ├── models/lgb_model.pkl          # separate LightGBM model from train_model.py
│   └── app/main.py                   # FastAPI app
└── README.md
```

The virtual environment for this project is the `customer_churn/` folder.

## Run the API

From `project/`, using that environment:

```bash
..\customer_churn\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

`POST /predict` with `{"data": [ {customer fields...}, ... ]}`. The response has `Churn_Pred` and `Churn_Prob`.

The API scores with `churn_stack_pipeline.pkl`. That pipeline already contains the scaler and one-hot encoder, so new rows keep a stable encoding. `project/models/lgb_model.pkl` is the separate LightGBM file produced by `train_model.py`.

