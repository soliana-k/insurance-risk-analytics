import pytest
import pandas as pd
from src.insurance_EDA import InsuranceEda   


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'TotalPremium': [1000.0, 2500.0, 1800.0, 3200.0],
        'TotalClaims': [200.0, 800.0, 150.0, 1200.0],
        'Province': ['Gauteng', 'Western Cape', 'Gauteng', 'KwaZulu-Natal'],
        'make': ['Toyota', 'BMW', 'Toyota', 'Volkswagen'],
        'Gender': ['Male', 'Female', 'Male', None]
    })


def test_initialization(sample_data):
    eda = InsuranceEda(sample_data)
    assert eda.df is not None
    assert len(eda.df) == 4


def test_loss_ratio_calculation(sample_data):
    eda = InsuranceEda(sample_data)
    total_p = eda.df['TotalPremium'].sum()
    total_c = eda.df['TotalClaims'].sum()
    loss_ratio = total_c / total_p
    assert 0 < loss_ratio < 1


def test_remediate_missing_values(sample_data):
    eda = InsuranceEda(sample_data)
    eda.remediate_missing_values()
    assert eda.df['Gender'].isna().sum() == 0   