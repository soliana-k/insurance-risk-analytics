import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class InsuranceHypothesisTester:
    """
    Automated statistical testing framework to validate risk and margin drivers 
    for ACIS underwriting optimization.
    """
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input DataFrame is empty. Framework execution aborted.")
        self.df = df.copy()
        self.df['Has_Claim'] = np.where(self.df['TotalClaims'] > 0, 1, 0)
        self.df['Net_Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

        title_map = {
            'Mr': 'Male',
            'Mrs': 'Female',
            'Ms': 'Female',
            'Miss': 'Female'
        }
        self.df['Gender_Proxy'] = self.df['Title'].map(title_map)

    def test_regional_frequency(self, group_a_prov: str, group_b_prov: str):
        subset = self.df[self.df['Province'].isin([group_a_prov, group_b_prov])].copy()
        contingency_table = pd.crosstab(subset['Province'], subset['Has_Claim'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        rate_a = subset[subset['Province'] == group_a_prov]['Has_Claim'].mean() * 100
        rate_b = subset[subset['Province'] == group_b_prov]['Has_Claim'].mean() * 100
        return p_value, rate_a, rate_b

    def test_zip_severity(self, group_a_zip: int, group_b_zip: int):
        mask_a = (self.df['PostalCode'] == group_a_zip) & (self.df['TotalClaims'] > 0)
        mask_b = (self.df['PostalCode'] == group_b_zip) & (self.df['TotalClaims'] > 0)
        data_a = self.df[mask_a]['TotalClaims'].dropna()
        data_b = self.df[mask_b]['TotalClaims'].dropna()
        if len(data_a) < 2 or len(data_b) < 2:
            return np.nan, np.nan, np.nan
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        return p_value, data_a.mean(), data_b.mean()

    def test_zip_margin(self, group_a_zip: int, group_b_zip: int):
        data_a = self.df[self.df['PostalCode'] == group_a_zip]['Net_Margin'].dropna()
        data_b = self.df[self.df['PostalCode'] == group_b_zip]['Net_Margin'].dropna()
        if len(data_a) < 2 or len(data_b) < 2:
            return np.nan, np.nan, np.nan
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        return p_value, data_a.mean(), data_b.mean()

    def test_proxy_gender_frequency(self, group_a_gender: str = 'Female', group_b_gender: str = 'Male'):
        """
        Test H0: There is no significant risk difference (Claim Frequency) between Women and Men.
        Uses the engineered Gender_Proxy feature to achieve full portfolio coverage.
        """
        
        subset = self.df[self.df['Gender_Proxy'].isin([group_a_gender, group_b_gender])].copy()
        
        total_valid_proxy = len(subset)
        portfolio_coverage = (total_valid_proxy / len(self.df)) * 100
        
        contingency_table = pd.crosstab(subset['Gender_Proxy'], subset['Has_Claim'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        rate_a = subset[subset['Gender_Proxy'] == group_a_gender]['Has_Claim'].mean() * 100
        rate_b = subset[subset['Gender_Proxy'] == group_b_gender]['Has_Claim'].mean() * 100
        
        return p_value, rate_a, rate_b, portfolio_coverage


def run_pipeline_orchestrator(dataframe):
    """
    Orchestrates the statistical testing using engineered proxy fields.
    """
    if dataframe.empty:
        print("Error: The provided notebook DataFrame is empty.")
        return

    tester = InsuranceHypothesisTester(dataframe)
    
    print("\n" + "="*70)
    print(" EXECUTIVE HYPOTHESIS TESTING STATISTICAL REPORT (RECONSTRUCTED) ")
    print("="*70 + "\n")

    # --- HYPOTHESIS 1: PROVINCES (Frequency) ---
    p1, r_a1, r_b1 = tester.test_regional_frequency('Western Cape', 'Gauteng')
    print("HYPOTHESIS 1: Regional Risk Differences (Claim Frequency)")
    print(f"  Target Segments : Control: Western Cape vs. Test: Gauteng")
    print(f"  Control Metric  : {r_a1:.4f}% Claim Frequency")
    print(f"  Test Metric     : {r_b1:.4f}% Claim Frequency")
    print(f"  Calculated p-val: {p1:.6e}")
    print(f"  Outcome         : {'REJECT H0' if p1 < 0.05 else 'FAIL TO REJECT H0'}")
    print("-" * 70)

    # --- HYPOTHESIS 2: ZIP CODES (Severity) ---
    zip_control, zip_test = 2000, 122
    p2, m_a2, m_b2 = tester.test_zip_severity(zip_control, zip_test)
    print("HYPOTHESIS 2: Geographic Risk Differences (Claim Severity)")
    print(f"  Target Segments : Control: Postal Code {zip_control} vs. Test: Postal Code {zip_test}")
    print(f"  Control Metric  : Average Severity of {m_a2:,.2f}" if not np.isnan(m_a2) else "  Control Metric  : No Claims")
    print(f"  Test Metric     : Average Severity of {m_b2:,.2f}" if not np.isnan(m_b2) else "  Test Metric     : No Claims")
    print(f"  Calculated p-val: {p2:.6e}" if not np.isnan(p2) else "  Calculated p-val: Insufficient Data")
    print(f"  Outcome         : {'REJECT H0' if p2 < 0.05 else 'FAIL TO REJECT H0'}")
    print("-" * 70)

    # --- HYPOTHESIS 3: ZIP CODES (Net Margin) ---
    p3, m_a3, m_b3 = tester.test_zip_margin(zip_control, zip_test)
    print("HYPOTHESIS 3: Geographic Performance Differences (Net Margin)")
    print(f"  Target Segments : Control: Postal Code {zip_control} vs. Test: Postal Code {zip_test}")
    print(f"  Control Metric  : Average Net Margin of {m_a3:,.2f}")
    print(f"  Test Metric     : Average Net Margin of {m_b3:,.2f}")
    print(f"  Calculated p-val: {p3:.6e}")
    print(f"  Outcome         : {'REJECT H0' if p3 < 0.05 else 'FAIL TO REJECT H0'}")
    print("-" * 70)

    # --- HYPOTHESIS 4: DEMOGRAPHIC RISK (Gender Proxy) ---
    p4, r_a4, r_b4, coverage = tester.test_proxy_gender_frequency('Female', 'Male')
    print("HYPOTHESIS 4: Demographic Risk Differences (Gender Claim Frequency)")
    print(f"  Target Segments : Control: Female vs. Test: Male")
    print(f"  Control Metric  : {r_a4:.4f}% Claim Frequency")
    print(f"  Test Metric     : {r_b4:.4f}% Claim Frequency")
    print(f"  Portfolio Coverage: {coverage:.2f}% of total data exposures (Excluding 'Dr')")
    print(f"  Calculated p-val: {p4:.6e}")
    print(f"  Outcome         : {'REJECT H0' if p4 < 0.05 else 'FAIL TO REJECT H0'}")
    print("=" * 70 + "\n")