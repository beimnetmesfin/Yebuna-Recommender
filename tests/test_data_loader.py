import pytest
import pandas as pd
from app.config import products_file,user_events_file
from app.data_loader import DataLoader
from pathlib import Path

#creates one reusable DataLoader for the tests instead of writing
#the same three lines repeatedly.
@pytest.fixture
def loader():
    return DataLoader(
        products_file,
        user_events_file
    )

def test_load_products(loader):
    df=loader.load_products()
    assert isinstance(df,pd.DataFrame)
    assert not df.empty
    assert "item_id" in df.columns

def test_load_user_events(loader):
    df=loader.load_user_events()

    assert isinstance(df,pd.DataFrame) 
    assert not df.empty
    assert "item_id" in df.columns
    assert "user_id" in df.columns

def test_missing_file(loader):
    loader=DataLoader(
        Path("data/not_existing.csv"),
        user_events_file
    )
    

    with pytest.raises(FileNotFoundError):
        loader.load_products()
        

             