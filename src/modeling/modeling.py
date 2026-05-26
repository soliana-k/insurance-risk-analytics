import time
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def train_frequency_models(X_train: pd.DataFrame, y_freq_train: pd.Series, X_test: pd.DataFrame):
    """
    Goal 1: Probability of a Claim Classifier Engine.
    Implements and tunes all three required architectures: Linear (LogReg), Random Forest, and XGBoost.
    """
    logger.info("="*60)
    logger.info("TASK 4: TRAINING ALL 3 REQUIRED CLASSIFICATION MODELS")
    logger.info("="*60)
    
    imbalance_ratio = (y_freq_train == 0).sum() / (y_freq_train == 1).sum()

    
    logger.info("Training Linear Classifier Baseline...")
    lr_clf = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1)
    lr_clf.fit(X_train, y_freq_train)
    lr_preds = lr_clf.predict(X_test)
    lr_prob = lr_clf.predict_proba(X_test)[:, 1]

    logger.info("Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_freq_train)
    rf_preds = rf_clf.predict(X_test)
    rf_prob = rf_clf.predict_proba(X_test)[:, 1]

 
    logger.info("Training XGBoost Classifier...")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=imbalance_ratio, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_freq_train)
    xgb_preds = xgb_clf.predict(X_test)
    xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]

    return (lr_preds, rf_preds, xgb_preds), (lr_prob, rf_prob, xgb_prob), (lr_clf, rf_clf, xgb_clf)


def train_severity_models(X_train: pd.DataFrame, y_sev_train: pd.Series, X_test: pd.DataFrame, y_sev_test: pd.Series):
    """
    Goal 2: Claim Severity Prediction Regressor Engine (Subset where claims > 0).
    Implements and tunes all three required architectures: Linear, Random Forest, and XGBoost.
    """
    logger.info("\n" + "="*60)
    logger.info("TASK 4: TRAINING ALL 3 REQUIRED REGRESSION MODELS")
    logger.info("="*60)

    train_claims_mask = y_sev_train > 0
    test_claims_mask = y_sev_test > 0

    X_train_sev = X_train[train_claims_mask]
    X_test_sev = X_test[test_claims_mask]

   
    y_train_sev_log = np.log1p(y_sev_train[train_claims_mask])
    y_test_sev_real = y_sev_test[test_claims_mask]


    logger.info("Training Linear Regression Baseline...")
    lr_reg = LinearRegression()
    lr_reg.fit(X_train_sev, y_train_sev_log)
    lr_reg_preds = np.expm1(lr_reg.predict(X_test_sev))

    
    logger.info("Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_sev, y_train_sev_log)
    rf_reg_preds = np.expm1(rf_reg.predict(X_test_sev))

    logger.info("Training XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train_sev, y_train_sev_log)
    xgb_reg_preds = np.expm1(xgb_reg.predict(X_test_sev))

    return (lr_reg_preds, rf_reg_preds, xgb_reg_preds), y_test_sev_real, (lr_reg, rf_reg, xgb_reg)


def generate_rubric_reports(y_freq_test, clf_preds, y_sev_test_real, reg_preds):
    """
    Generates exact evaluation comparison metrics side-by-side as dictated by task goals.
    """
    lr_preds_c, rf_preds_c, xgb_preds_c = clf_preds
    lr_preds_r, rf_preds_r, xgb_preds_r = reg_preds

    print("\n" + "═"*60 + "\nCLASSIFICATION COMPARISON (FREQUENCY SELECTION)\n" + "═"*60)
    clf_summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Linear Regression (LogReg)": [
            accuracy_score(y_freq_test, lr_preds_c),
            precision_score(y_freq_test, lr_preds_c, zero_division=0),
            recall_score(y_freq_test, lr_preds_c, zero_division=0),
            f1_score(y_freq_test, lr_preds_c, zero_division=0)
        ],
        "Random Forest Classifier": [
            accuracy_score(y_freq_test, rf_preds_c),
            precision_score(y_freq_test, rf_preds_c, zero_division=0),
            recall_score(y_freq_test, rf_preds_c, zero_division=0),
            f1_score(y_freq_test, rf_preds_c, zero_division=0)
        ],
        "XGBoost Classifier": [
            accuracy_score(y_freq_test, xgb_preds_c),
            precision_score(y_freq_test, xgb_preds_c, zero_division=0),
            recall_score(y_freq_test, xgb_preds_c, zero_division=0),
            f1_score(y_freq_test, xgb_preds_c, zero_division=0)
        ]
    }).set_index("Metric")
    print(clf_summary.round(4).to_string())

    print("\n" + "═"*60 + "\nREGRESSION COMPARISON (SEVERITY PAYOUTS)\n" + "═"*60)
    reg_summary = pd.DataFrame({
        "Metric": ["RMSE (Lower = Better)", "R² Score (Higher = Better)"],
        "Linear Regression Baseline": [
            np.sqrt(mean_squared_error(y_sev_test_real, lr_preds_r)),
            r2_score(y_sev_test_real, lr_preds_r)
        ],
        "Random Forest Regressor": [
            np.sqrt(mean_squared_error(y_sev_test_real, rf_preds_r)),
            r2_score(y_sev_test_real, rf_preds_r)
        ],
        "XGBoost Regressor Engine": [
            np.sqrt(mean_squared_error(y_sev_test_real, xgb_preds_r)),
            r2_score(y_sev_test_real, xgb_preds_r)
        ]
    }).set_index("Metric")
    print(reg_summary.round(2).to_string())