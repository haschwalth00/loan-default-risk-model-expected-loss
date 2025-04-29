# Loan Default Risk Model & Expected Loss Estimator

## Overview

This independent machine learning project focuses on building a predictive model to assess the **probability of default (PD)** and estimate the **expected loss (EL)** for borrowers based on key financial attributes. It combines classification models with financial logic (LGD × EAD × PD) to evaluate individual loan risk.

## Author

**Vanshwardhan Singh**

## Objective

To simulate a credit risk assessment framework that:
- Predicts a borrower's **likelihood of default** using supervised ML techniques
- Calculates **expected monetary loss** based on loan characteristics and an assumed loss-given-default (LGD)
- Provides a callable Python function for scenario testing with new borrower profiles

## Key Features

- Trains two classifiers:  
  - **Logistic Regression** for probabilistic linear modeling  
  - **Decision Tree Classifier** for interpretable rule-based modeling

- Calculates:
  - `Probability of Default (PD)` from the selected model
  - `Expected Loss (EL) = PD × LGD × Exposure at Default`

- Easy-to-use function:  
  `predict_expected_loss(borrower_features, model='logistic')`

## Dataset

- Source: Synthetic dataset simulating loan-level borrower information
- Columns used:
  - `credit_lines_outstanding`
  - `loan_amt_outstanding`
  - `total_debt_outstanding`
  - `income`
  - `years_employed`
  - `fico_score`
  - `default` (label)

## Example Input

```python
example_borrower = {
    'credit_lines_outstanding': 3,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 10000,
    'income': 60000,
    'years_employed': 5,
    'fico_score': 700
}
