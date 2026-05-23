# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import logging
# import os 


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# class InsuranceEda:
#     def __init__(self, df: pd.DataFrame):
#         self.df=df.copy()
        

#     def descriptive_stats(self):
#         logger.info('Descriptive statistics and Data summarization in process')
#         print(self.df.describe().T)
#         print(self.df.info())

#     def data_quality(self):
#         logger.info('Doing data quality check in progress')
#         missing_data = self.df.isna().mean() * 100
#         print(missing_data.sort_values(ascending=False))
#         cols_to_drop = missing_data[missing_data > 50].index.tolist()
#         print(f"\nColumns where more than 50% data is missing are:\n {cols_to_drop}")

#         logger.info('stating data type check and dynamic category conversion')
#         nons = ['CustomValueEstimate', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet']
#         columns = [col for col in self.df.columns if col not in nons]
        
#         string_columns = self.df[columns].select_dtypes(include=[str]).columns

#         for column in string_columns:
#             logger.info(f'Starting data check for {column}')
#             opt = self.df[column].value_counts()
#             unique_count = len(opt)
#             print(f'Possible {unique_count} options for {column}')
            
#             # --- Dynamic Type Conversion Based on Cardinality ---
#             if unique_count <= 1:
#                 logger.warning(f"-> Feature '{column}' has ZERO variance (only 1 unique option). Recommend dropping.")
#             else:
#                 # Safely convert all multi-option strings (both low and high cardinality like Model) to category
#                 self.df[column] = self.df[column].astype('category')
#                 logger.info(f"-> Successfully converted '{column}' from str to category type.")
                
#         print("\n=== Data quality check and structural type casting complete ===")

        
#         # pass
    
#     def assess_and_remediate(self) -> pd.DataFrame:
#         """
#         Executes the documented Data Quality Assessment handling strategy 
#         optimized for insurance risk analysis.
#         """
#         cleaned_df = self.df.copy()
#         cleaned_df['CapitalOutstanding'] = cleaned_df['CapitalOutstanding'].fillna('0')
#         # ---Drop Zero-Variance/Unsalvageable Features (>99% Missing) ---
#         structural_drops = ['NumberOfVehiclesInFleet', 'CrossBorder']
#         cleaned_df.drop(columns=structural_drops, errors='ignore', inplace=True)
        
#         # --Map Implicit Booleans (Systemic NaNs = Clean Record) ---
#         implicit_boolean_cols = ['WrittenOff', 'Rebuilt', 'Converted']
#         for col in implicit_boolean_cols:
#             if col in cleaned_df.columns:
#                 cleaned_df[col] = cleaned_df[col].notna().astype(int)
                
#         # --Feature Engineer & Impute Custom Valuations ---
#         if 'CustomValueEstimate' in cleaned_df.columns:
#             cleaned_df['Has_Custom_Estimate'] = cleaned_df['CustomValueEstimate'].notna().astype(int)
#             cleaned_df['CustomValueEstimate'] = cleaned_df['CustomValueEstimate'].fillna(0)
            
#         # -- Medium Missingness - Placeholder Encoding (~15% Missing) ---
#         if 'Bank' in cleaned_df.columns:
#             cleaned_df['Bank'] = cleaned_df['Bank'].fillna('Unfinanced_Or_Unknown')
#         if 'NewVehicle' in self.df.columns:
#             cleaned_df['NewVehicle'] = cleaned_df['NewVehicle'].fillna('Unknown')
            
#         # --Low Missingness - Statistical Imputation (<5% Missing) ---
#         demographic_cols = ['AccountType', 'Gender', 'MaritalStatus']
#         for col in demographic_cols:
#             if col in cleaned_df.columns:
#                 mode_value = cleaned_df[col].mode()[0]
#                 cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                
#         # --- Tier 6: Systemic Lookup Failures (0.055% Rows) ---
#         # Drop rows where the critical vehicle identity cannot be established
#         critical_identity_cols = ['make', 'Model', 'mmcode']
#         cleaned_df.dropna(subset=critical_identity_cols, axis=0, inplace=True)
#         print('the assesment and cleanup is done')
#         print(cleaned_df.info())
#         return cleaned_df

#     def univaraite_analysis(self):
#         nons=['CustomValueEstimate','WrittenOff','Rebuilt','Converted','CrossBorder','NumberOfVehiclesInFleet']
#         columns=[col for col in self.df.columns if col not in nons]
#         for column in columns:
#             if self.df[column].dtype.kind in ('i', 'f'):
#                 plt.figure() 
#                 sns.histplot(self.df[column], kde=True)
#                 plt.title(f'Histogram of {column}')
#                 plt.xlabel(column)
#                 plt.ylabel('Frequency')
#                 plt.show() 
            

#     def multi(self):
#         pass
#     def run(self):
#         self.descriptive_stats()
#         self.assess_and_remediate()
#         self.data_quality()
#         self.univaraite_analysis()
        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsuranceEda:
    def __init__(self, df: pd.DataFrame):
        
        self.df = df.copy()
        
    def descriptive_stats(self):
        logger.info('--- STEP 1: Descriptive Statistics & Data Summarization ---')
        print("\n[Numerical Feature Summary]")
        print(self.df.describe().T)
        print("\n[DataFrame Structural Info]")
        print(self.df.info())

    def data_quality_assessment(self):
        logger.info('--- STEP 2: Data Quality & Structural Audit ---')
        
        # 1. Check Missingness
        missing_data = self.df.isna().mean() * 100
        print("\n[Missing Data Percentages per Column]")
        print(missing_data.sort_values(ascending=False))
        
        cols_to_drop = missing_data[missing_data > 50].index.tolist()
        print(f"\nColumns where more than 50% data is missing: {cols_to_drop}")

        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
        if 'VehicleIntroDate' in self.df.columns:
            self.df['VehicleIntroDate'] = pd.to_datetime(self.df['VehicleIntroDate'], errors='coerce')
        if 'CapitalOutstanding' in self.df.columns:
            self.df['CapitalOutstanding'] = self.df['CapitalOutstanding'].astype(str).str.replace(r'[$,\s]', '', regex=True)
            self.df['CapitalOutstanding'] = pd.to_numeric(self.df['CapitalOutstanding'], errors='coerce').fillna(0)

    def remediate_missing_values(self):
        logger.info('--- STEP 3: Executing Insurance-Optimized Data Remediation ---')
        
        
        structural_drops = ['NumberOfVehiclesInFleet', 'CrossBorder', 'Language', 'Country', 'ItemType', 'StatutoryClass', 'StatutoryRiskType']
        self.df.drop(columns=structural_drops, errors='ignore', inplace=True)
        
      
        implicit_boolean_cols = ['WrittenOff', 'Rebuilt', 'Converted']
        for col in implicit_boolean_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].notna().astype(int)
                
        # 3. Handle Custom Valuations
        if 'CustomValueEstimate' in self.df.columns:
            self.df['Has_Custom_Estimate'] = self.df['CustomValueEstimate'].notna().astype(int)
            self.df['CustomValueEstimate'] = self.df['CustomValueEstimate'].fillna(0)
            
        # 4. FIRST: Fill text placeholders WHILE they are still standard string columns
        if 'Bank' in self.df.columns:
            self.df['Bank'] = self.df['Bank'].fillna('Unfinanced_Or_Unknown')
        if 'NewVehicle' in self.df.columns:
            self.df['NewVehicle'] = self.df['NewVehicle'].fillna('Unknown')
            
        # 5. Low Missingness - Statistical Imputation
        demographic_cols = ['AccountType', 'Gender', 'MaritalStatus']
        for col in demographic_cols:
            if col in self.df.columns:
                mode_value = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_value)
                
        # 6. Tier 6: Systemic Lookup Failures (Drop unidentifiable vehicles)
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
        
        logger.info('Assessment, cleanup, and type adjustments are fully committed.')

    def univariate_analysis(self):
        logger.info('--- STEP 4: Generating Univariate Visualizations ---')
        
        
        sns.set_theme(style="whitegrid", palette="muted")
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = self.df.select_dtypes(include=['category']).columns
        
        # 1. Numerical Plots: Histograms
        print("\nRendering numerical features...")
        for column in num_cols:
            if self.df[column].nunique() <= 2:
                continue
                
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[column], kde=True, bins=40)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show() 

        # 2. Categorical Plots: Bar Charts
        print("\nRendering categorical features...")
        for column in cat_cols:
            unique_count = self.df[column].nunique()
            
            plt.figure(figsize=(10, 4))
            if unique_count > 25:
                order = self.df[column].value_counts().iloc[:10].index
                sns.countplot(data=self.df, y=column, order=order)
                plt.title(f'Top 10 Values for High-Cardinality Feature: {column}')
            else:
                order = self.df[column].value_counts().index
                ax=sns.countplot(data=self.df, x=column, order=order)
                ax.set_yscale('log')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {column}')
                
            plt.tight_layout()
            plt.show()

    def run(self):
        """
        Main pipeline orchestrator. Mutates state chronologically 
        so data is summarized, audited, cleaned, and then correctly plotted.
        """
        self.descriptive_stats()           
        self.data_quality_assessment()     
        self.remediate_missing_values()    
        self.descriptive_stats()          
        self.univariate_analysis()         
        return self.df                     