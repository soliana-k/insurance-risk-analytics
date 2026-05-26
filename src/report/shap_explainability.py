import os
import logging
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def generate_shap_analysis(model, X_data: pd.DataFrame, output_dir: str = "../reports/figures/"):
    """
    Calculates SHAP values for tree-based architectures, saves a high-resolution 
    summary plot, and returns a structured table ranking features by mean absolute impact.
    """
    logger.info("Initializing SHAP Interpretability Engine...")
    
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shap_severity_summary.png")
    
    X_sample = X_data.sample(min(2000, len(X_data)), random_state=42)
    
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    

    shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
    
    plt.title("Top 10 Feature Drivers: Claim Severity Risk Model", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"SHAP summary plot successfully exported to: {output_path}")
    
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    feature_importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    
    print("\n" + "═"*60 + "\nTOP 10 SHAP FEATURE IMPORTANCE SELECTIONS\n" + "═"*60)
    print(feature_importance_df.head(10).to_string(index=False))
    
    return feature_importance_df


