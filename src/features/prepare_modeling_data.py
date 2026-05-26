import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data_and_split(file_path: str, test_size: float = 0.3):
    """
    Executes data preparation using scikit-learn's OneHotEncoder to transform 
    categorical variables, preventing data leakage across train/test splits.
    """
    logger.info("Loading raw dataset for comprehensive preparation...")
    df = pd.read_csv(file_path, sep='|', low_memory=False)
    
  
    # STEP 1: IMPUTATION
    implicit_flags = ['WrittenOff', 'Rebuilt', 'Converted']
    for col in implicit_flags:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)
            
    if 'NumberOfVehiclesInFleet' in df.columns:
        df['NumberOfVehiclesInFleet'] = df['NumberOfVehiclesInFleet'].fillna(0)

    missing_ratios = df.isna().mean() * 100
    cols_above_30 = missing_ratios[missing_ratios > 30].index.tolist()
    cols_to_drop = [c for c in cols_above_30 if c not in ['Gender', 'Title']]
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    
    # STEP 2: FEATURE ENGINEERING
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
    
    df['Vehicle_Age'] = df['TransactionMonth'].dt.year - df['VehicleIntroDate'].dt.year
    df['Vehicle_Age'] = df['Vehicle_Age'].apply(lambda x: x if x >= 0 else np.nan).fillna(df['Vehicle_Age'].median())

    possible_start_cols = ['CoverStartDate', 'PolicyStartDate', 'RegistrationDate']
    start_col = next((c for c in possible_start_cols if c in df.columns), None)
    if start_col:
        df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        df['Policy_Duration'] = (df['TransactionMonth'] - df[start_col]).dt.days
    else:
        df['Policy_Duration'] = 0
    df['Policy_Duration'] = df['Policy_Duration'].apply(lambda x: x if x >= 0 else 0).fillna(df['Policy_Duration'].median())
    
    title_gender_map = {'Mr': 'Male', 'Mrs': 'Female', 'Ms': 'Female', 'Miss': 'Female'}
    df['Gender_Cleaned'] = df['Gender'].fillna(df['Title'].map(title_gender_map)).fillna('Unspecified')
    
    title_marital_map = {'Mrs': 'Married', 'Miss': 'Single'}
    df['MaritalStatus_Cleaned'] = df['MaritalStatus'].fillna(df['Title'].map(title_marital_map)).fillna('Unknown')
    
    top_12_makes = df['make'].value_counts().index[:12]
    df['Make_Cleaned'] = df['make'].apply(lambda x: x if x in top_12_makes else 'Other')
    
    df['Has_Claim'] = np.where(df['TotalClaims'] > 0, 1, 0)
    
   
    # STEP 3: SEPARATION OF NUMERICAL VS CATEGORICAL
    num_features = ['Vehicle_Age', 'Policy_Duration', 'SumInsured', 'WrittenOff', 'NumberOfVehiclesInFleet']
    cat_features = ['VehicleType', 'Bank', 'Gender_Cleaned', 'MaritalStatus_Cleaned', 'Make_Cleaned', 'Province', 'CoverType', 'CoverGroup', 'ExcessSelected']
    
    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]


    logger.info("Executing scikit-learn OneHotEncoder transformation...")
    
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error')
    encoded_cat_array = encoder.fit_transform(df[cat_features])
    
    encoded_cat_df = pd.DataFrame(
        encoded_cat_array, 
        columns=encoder.get_feature_names_out(cat_features),
        index=df.index
    )
    
    X = pd.concat([df[num_features], encoded_cat_df], axis=1)
    
    y_severity = df['TotalClaims'].copy()
    y_frequency = df['Has_Claim'].copy()
    
   
    logger.info(f"Splitting arrays into train/test sets with a 70:30 ratio...")
    X_train, X_test, y_freq_train, y_freq_test = train_test_split(
        X, y_frequency, test_size=test_size, random_state=42, stratify=y_frequency
    )
    
    y_sev_train = y_severity.loc[X_train.index]
    y_sev_test = y_severity.loc[X_test.index]
    
    logger.info(f"Data preparation complete via sklearn. Final Matrix Feature Dimensions: {X_train.shape[1]}")
    return X_train, X_test, y_freq_train, y_freq_test, y_sev_train, y_sev_test


