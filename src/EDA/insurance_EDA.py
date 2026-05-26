import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsuranceEda:
    """
    A modular class for performing Exploratory Data Analysis on AlphaCare Insurance claim data.
    
    This class handles data quality assessment, visualization, and KPI calculation to support
    risk segmentation and pricing strategy decisions.
    
    Attributes:
        df (pd.DataFrame): The insurance dataset being analyzed.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a copy of the input DataFrame to avoid modifying original data.
        
        Args:
            df (pd.DataFrame): Raw insurance dataset.
        """
        self.df = df.copy()
        
    def descriptive_stats(self)-> None:
        """Generate descriptive statistics and structural information for the dataset."""
        logger.info('--- STEP 1: Descriptive Statistics & Data Summarization ---')
        print("\n[Numerical Feature Summary]")
        print(self.df.describe().T)
        print("\n[DataFrame Structural Info]")
        print(self.df.info())

    def data_quality_assessment(self)-> None:
        """Assess data quality, missing values, and perform initial type conversions."""
        logger.info('--- STEP 2: Data Quality & Structural Audit ---')
        
        # 1. Check Missingness
        missing_data = self.df.isna().mean() * 100
        print("\n[Missing Data Percentages per Column]")
        print(missing_data.sort_values(ascending=False))
        
        cols_to_drop = missing_data[missing_data > 50].index.tolist()
        print(f"\nColumns where more than 50% data is missing: {cols_to_drop}")

        date_cols = ['TransactionMonth', 'VehicleIntroDate']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        if 'CapitalOutstanding' in self.df.columns:
            self.df['CapitalOutstanding'] = self.df['CapitalOutstanding'].astype(str).str.replace(r'[$,\s]', '', regex=True)
            self.df['CapitalOutstanding'] = pd.to_numeric(self.df['CapitalOutstanding'], errors='coerce').fillna(0)

    def remediate_missing_values(self)-> None:
        """Apply insurance-specific data remediation and type casting."""
        logger.info('--- STEP 3: Executing Insurance-Optimized Data Remediation ---')
        try:
            structural_drops = ['NumberOfVehiclesInFleet', 'CrossBorder', 'Language', 'Country', 'ItemType', 'StatutoryClass', 'StatutoryRiskType']
            self.df.drop(columns=structural_drops, errors='ignore', inplace=True)
        
            implicit_boolean_cols = ['WrittenOff', 'Rebuilt', 'Converted']
            for col in implicit_boolean_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].notna().astype(int)
                    
            if 'CustomValueEstimate' in self.df.columns:
                self.df['Has_Custom_Estimate'] = self.df['CustomValueEstimate'].notna().astype(int)
                self.df['CustomValueEstimate'] = self.df['CustomValueEstimate'].fillna(0)
                
            if 'Bank' in self.df.columns:
                self.df['Bank'] = self.df['Bank'].fillna('Unfinanced_Or_Unknown')
            if 'NewVehicle' in self.df.columns:
                self.df['NewVehicle'] = self.df['NewVehicle'].fillna('Unknown')
                
            demographic_cols = ['AccountType', 'Gender', 'MaritalStatus']
            for col in demographic_cols:
                if col in self.df.columns:
                    mode_value = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    
            critical_identity_cols = ['make', 'Model', 'mmcode']
            self.df.dropna(subset=critical_identity_cols, axis=0, inplace=True)
            
            nons = ['CustomValueEstimate', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet', 'TransactionMonth', 'VehicleIntroDate', 'CapitalOutstanding', 'Has_Custom_Estimate']
            string_columns = self.df.drop(columns=nons, errors='ignore').select_dtypes(include=['object', 'string']).columns

            print("\n[String Column Cardinality Scan & Category Conversion]")
            for column in string_columns:
                unique_count = self.df[column].nunique()
                print(f"Column: {column:22} | Unique Options: {unique_count}")
                
                if unique_count <= 1:
                    logger.warning(f"-> '{column}' has ZERO variance. Stays as object (flagged for dropping).")
                else:
                    self.df[column] = self.df[column].astype('category')
                    logger.info(f"-> Successfully cast '{column}' to category type.")
        except Exception as e:
            logger.error(f"Unexpected error encountered during step 3: {str(e)}")
            raise
         
        logger.info('Assessment, cleanup, and type adjustments are fully committed.')

    def univariate_analysis(self)-> None:
        """Generate univariate visualizations for numerical and categorical features."""
        logger.info('--- STEP 4: Generating Univariate Visualizations ---')
        sns.set_theme(style="whitegrid", palette="muted")
         
        print("\nRendering numerical features...")
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for column in num_cols:
            if self.df[column].nunique() <= 2:
                continue
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[column], kde=True, bins=40)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
            plt.close()

        print("\nRendering categorical features...")
        cat_cols = self.df.select_dtypes(include=['category']).columns
        for column in cat_cols:
            unique_count = self.df[column].nunique()
            plt.figure(figsize=(10, 4))
            if unique_count > 25:
                order = self.df[column].value_counts().iloc[:10].index
                sns.countplot(data=self.df, y=column, order=order)
                plt.title(f'Top 10 Values for {column}')
            else:
                order = self.df[column].value_counts().index
                ax = sns.countplot(data=self.df, x=column, order=order)
                ax.set_yscale('log')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()
            plt.close()

    def bivariate_multivariate_analysis(self)-> None:
        """Analyze relationships between financial variables and geographic performance."""
        logger.info('--- STEP 5: Executing Bivariate & Multivariate Analysis ---')
        analysis_df = self.df.copy()
        analysis_df['TotalPremium'] = pd.to_numeric(analysis_df['TotalPremium'], errors='coerce')
        analysis_df['TotalClaims'] = pd.to_numeric(analysis_df['TotalClaims'], errors='coerce')
         
        financial_cols = ['TotalPremium', 'TotalClaims', 'CapitalOutstanding', 'CustomValueEstimate']
        existing_cols = [col for col in financial_cols if col in analysis_df.columns]
         
        corr_matrix = analysis_df[existing_cols].corr()
        print("\n[Financial Feature Correlation]")
        print(corr_matrix)
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
        plt.title('Financial Correlation Matrix')
        plt.tight_layout()
        plt.show()

        if 'PostalCode' in analysis_df.columns:
            logger.info('Aggregating performance metrics by ZipCode...')
            zip_agg = analysis_df.groupby('PostalCode').agg(
                Total_Premium=('TotalPremium', 'sum'),
                Total_Claims=('TotalClaims', 'sum'),
                Policy_Count=('TotalPremium', 'count')
            ).dropna().reset_index()
             
            zip_agg['Loss_Ratio'] = zip_agg['Total_Claims'] / (zip_agg['Total_Premium'] + 1e-5)
            zip_agg['Margin'] = zip_agg['Total_Premium'] - zip_agg['Total_Claims']
             
            plt.figure(figsize=(10, 6))
            q_prem = zip_agg['Total_Premium'].quantile(0.98)
            q_claim = zip_agg['Total_Claims'].quantile(0.98)
            plot_data = zip_agg[(zip_agg['Total_Premium'] <= q_prem) & (zip_agg['Total_Claims'] <= q_claim)]
            
            scatter = plt.scatter(
                x=plot_data['Total_Premium'], 
                y=plot_data['Total_Claims'], 
                c=plot_data['Margin'],       
                cmap='RdYlGn',                
                alpha=0.6, 
                edgecolors='w',
                s=plot_data['Policy_Count'] / 10 + 20
            )
            plt.colorbar(scatter, label='Net Margin (Premium - Claims)')
            plt.title('PostalCode Financial Mapping: Total Premium vs Total Claims')
            plt.xlabel('Total Premium Volume')
            plt.ylabel('Total Claim Payouts')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    def geographic_trends(self)-> None:
        """Analyze geographic and vehicle make patterns across provinces."""
        logger.info('--- STEP 6: Analyzing Geographic Trends across Provinces ---')
        geo_df = self.df.copy()
        if 'Province' in geo_df.columns and 'TotalPremium' in geo_df.columns:
            geo_df['TotalPremium'] = pd.to_numeric(geo_df['TotalPremium'], errors='coerce')
            plt.figure(figsize=(12, 5))
            prov_premium = geo_df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False).reset_index()
            sns.barplot(data=prov_premium, x='Province', y='TotalPremium', palette='viridis')
            plt.title('Average Total Premium per Province')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        # COVER TYPE ACROSS PROVINCES
        if 'CoverType' in geo_df.columns:
            plt.figure(figsize=(12, 6))
            cover_prov = pd.crosstab(geo_df['Province'], geo_df['CoverType'], normalize='index') * 100
            cover_prov.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='tab10', edgecolor='white')
            plt.title('Proportional Distribution of Cover Type across Provinces')
            plt.xlabel('Province')
            plt.ylabel('Percentage Share (%)')
            plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        if 'make' in geo_df.columns:
            print("\n[Top 5 Auto Makes across Selected High-Volume Provinces]")
            top_provinces = geo_df['Province'].value_counts().iloc[:4].index
            
            for prov in top_provinces:
                prov_subset = geo_df[geo_df['Province'] == prov]
                top_makes = prov_subset['make'].value_counts(normalize=True).head(5) * 100
                print(f"\n--- Province: {prov} ---")
                for make, share in top_makes.items():
                    print(f"  {make:15}: {share:.2f}% of regional vehicle pool")

    def outlier_detection(self)-> None:
        """Detect and visualize outliers in key financial variables."""
        logger.info('--- STEP 7: Executing Outlier Detection via Statistical Box Plots ---')
        outlier_df = self.df.copy()
        key_numerical_features = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
         
        for feature in key_numerical_features:
            if feature in outlier_df.columns:
                outlier_df[feature] = pd.to_numeric(outlier_df[feature], errors='coerce')
                q1 = outlier_df[feature].quantile(0.25)
                q3 = outlier_df[feature].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + (1.5 * iqr)
                 
                total_outliers = (outlier_df[feature] > upper_bound).sum()
                print(f"[{feature}] Outliers detected: {total_outliers}")
                pct_outliers = (total_outliers / len(outlier_df)) * 100
                
                print(f"\n[{feature} Outlier Metrics]")
                print(f"  IQR: {iqr:.2f} | Upper Whisker Limit: {upper_bound:.2f}")
                print(f"  Total Outlying Anomalies detected: {total_outliers} ({pct_outliers:.3f}%)")
                
                plt.figure(figsize=(10, 3))
                plot_data = outlier_df[outlier_df[feature] > 0]
                ax = sns.boxplot(data=plot_data, x=feature, color='lightblue')
                ax.set_xscale('log')
                plt.title(f'Log-Scaled Box Plot: {feature}')
                plt.tight_layout()
                plt.show()


    def analyze_temporal_trends(self)-> None:
        """Analyze claim frequency and severity trends over the 18-month period."""
        logger.info('--- STEP 7.5: Analyzing Temporal Trends over 18 Months ---')
        temporal_df = self.df.copy()
        
        if 'TransactionMonth' in temporal_df.columns and 'TotalClaims' in temporal_df.columns:
            
            temporal_df['TransactionMonth'] = pd.to_datetime(temporal_df['TransactionMonth'])
        
            monthly_trends = temporal_df.groupby('TransactionMonth').agg(
                Total_Claims_Paid=('TotalClaims', 'sum'),
                Total_Claim_Incidents=('TotalClaims', lambda x: (x > 0).sum()),
                Average_Claim_Severity=('TotalClaims', lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0),
                Total_Policies_Active=('TotalPremium', 'count')
            ).sort_index().reset_index()
            
        
            monthly_trends['Claim_Frequency_Rate'] = (monthly_trends['Total_Claim_Incidents'] / monthly_trends['Total_Policies_Active']) * 100
            
            print("\n[18-Month Timeline Trend Metrics Summary]")
            print(monthly_trends[['TransactionMonth', 'Claim_Frequency_Rate', 'Average_Claim_Severity']])
            
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            # Left Axis: Claim Frequency
            color = 'tab:blue'
            ax1.set_xlabel('Transaction Timeline (Months)')
            ax1.set_ylabel('Claim Frequency Rate (%)', color=color)
            sns.lineplot(data=monthly_trends, x='TransactionMonth', y='Claim_Frequency_Rate', ax=ax1, color=color, marker='o', linewidth=2.5)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Right Axis: Claim Severity
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Average Claim Severity (Currency units)', color=color)
            sns.lineplot(data=monthly_trends, x='TransactionMonth', y='Average_Claim_Severity', ax=ax2, color=color, marker='s', linewidth=2.5)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Insurance Timeline Analysis: Claim Frequency vs Severity Trends')
            fig.tight_layout()
            plt.show()
            plt.close()
        else:
            logger.warning("Missing 'TransactionMonth' or 'TotalClaims' columns. Skipping timeline trend run.")

    def calculate_business_metrics(self)-> None:
        """Calculate core insurance KPIs including Loss Ratio and segment-level insights."""
        logger.info('--- STEP 8: Calculating Core KPI Performance Metrics ---')
        
        total_claims = self.df['TotalClaims'].sum()
        total_premium = self.df['TotalPremium'].sum()
        overall_loss_ratio = total_claims / total_premium
        
        print("\n=======================================================")
        print(f"OVERALL LOSS RATIO: {overall_loss_ratio:.4f} ({overall_loss_ratio * 100:.2f}%)")
        print("=======================================================\n")
        
        segmentation_targets = ['Province', 'VehicleType', 'Gender']
        for target in segmentation_targets:
            if target in self.df.columns:
                print(f"[Loss Ratio Variances Across: {target}]")
                metrics = self.df.groupby(target, observed=True).agg(
                    Total_Claims=('TotalClaims', 'sum'),
                    Total_Premium=('TotalPremium', 'sum')
                )
                metrics['Loss_Ratio'] = metrics['Total_Claims'] / metrics['Total_Premium']
                print(metrics['Loss_Ratio'].sort_values(ascending=False))
                print("-" * 50)
        
        make_stats = self.df.groupby('make')['TotalClaims'].mean().reset_index()
        active_claims = make_stats[make_stats['TotalClaims'] > 0].sort_values(by='TotalClaims', ascending=False)

        print("=== VEHICLE MAKES WITH THE HIGHEST AVERAGE CLAIMS ===")
        print(active_claims.head(5).to_string(index=False, formatters={'TotalClaims': '{:,.2f}'.format}))

        print("\n=== VEHICLE MAKES WITH THE LOWEST AVERAGE CLAIMS ===")
        print(active_claims.tail(5).to_string(index=False, formatters={'TotalClaims': '{:,.2f}'.format}))

    def run(self) -> pd.DataFrame:
        """
        Execute the full EDA pipeline in sequence.
        
        Returns:
            pd.DataFrame: The cleaned and processed dataset.
        """
        self.descriptive_stats()           
        self.data_quality_assessment()
        self.remediate_missing_values()       
        self.univariate_analysis()    
        self.bivariate_multivariate_analysis()
        self.geographic_trends()
        self.outlier_detection()
        self.analyze_temporal_trends()
        self.calculate_business_metrics()

        path='../data/processed/cleaned_insurance_data.csv'
        self.df.to_csv(path, index=False)
        print(f"Cleaned data saved to {path}")