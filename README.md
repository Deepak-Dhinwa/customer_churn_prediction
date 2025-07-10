# Customer Churn Prediction

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

customer_churn/
│
├── app/
│ ├── train.py # Model training script
│ ├── predict.py # FastAPI prediction script
│ └── data_processing.py # Data cleaning and feature engineering utils
├── customer_churn.ipynb # Full EDA, preprocessing, modeling, evaluation
├── churn_model.pkl # Trained model weights
├── requirements.txt # Dependencies
└── README.md

---

## Minimal Usage

## Install dependencies

```bash
pip install -r requirements.txt
uvicorn app.predict:app --reload
```

## Make predictions

Send a POST request with customer data to the /predict endpoint. Receive churn probability and classification for operational deployment or CRM integration.

