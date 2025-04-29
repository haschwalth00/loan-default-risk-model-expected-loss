
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.interpolate import interp1d
from datetime import datetime

# Load loan data
loan_data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Prepare features and labels
X = loan_data.drop(columns=['customer_id', 'default'])
y = loan_data['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Train Decision Tree Model
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Recovery rate (assumed)
LGD = 0.9

# Prediction function
def predict_expected_loss(borrower_features, model='logistic'):
    """
    borrower_features: dict with keys matching feature names:
        'credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding',
        'income', 'years_employed', 'fico_score'
    
    model: 'logistic' or 'tree' (which model to use)
    """
    input_df = pd.DataFrame([borrower_features])
    
    if model == 'logistic':
        pd_pred = log_reg.predict_proba(input_df)[0, 1]
    elif model == 'tree':
        pd_pred = tree.predict_proba(input_df)[0, 1]
    else:
        raise ValueError("Model must be either 'logistic' or 'tree'.")
    
    expected_loss = pd_pred * LGD * borrower_features['loan_amt_outstanding']
    
    return {
        'probability_of_default': round(pd_pred, 4),
        'expected_loss': round(expected_loss, 2)
    }

# Example usage
if __name__ == "__main__":
    example_borrower = {
        'credit_lines_outstanding': 3,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 10000,
        'income': 60000,
        'years_employed': 5,
        'fico_score': 700
    }

    prediction = predict_expected_loss(example_borrower, model='logistic')
    print(f"Prediction: {prediction}")
