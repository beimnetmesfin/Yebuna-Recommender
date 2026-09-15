import pytest
import pandas as pd
from app.text_builder import semantic_columns,create_text_embedding,build_text_embedding

def test_create_text_embedding():
    row=pd.Series(
        {
        "title":"Ethiopia Coffee",
        "description":"Premuim Arabica coffee",
        "category":"Food",
        "subcategory":"coffee",
        "tags":"Arabic,organic",
        "creator_name":"yebuna",
        "language":"English",
        "country":"Ethiopia",
        }
    )

    text=create_text_embedding(row)

    assert "Ethiopia Coffee" in text
    assert "Premuim Arabica coffee" in text
    assert "Food" in text
    assert "coffee" in text
    assert "Ethiopia" in text

def test_build_text_embedding():
    df=pd.DataFrame(
        [
            {
           "title":"Ethiopia Coffee",
           "description":"Premuim Arabica coffee",
           "category":"Food",
           "subcategory":"coffee",
           "tags":"Arabic,organic",
           "creator_name":"yebuna",
           "language":"English",
           "country":"Ethiopia",     
            },
        {
        "title":"Ethiopia Basket",
        "description":"ethioian handmade",
        "category":"crafts",
        "subcategory":"Basket",
        "tags":"handmade,tradational",
        "creator_name":"local_creator",
        "language":"ameharic",
        "country":"Ethiopia",     
          },
    ]
    )
    texts=build_text_embedding(df)
    assert len(texts)==2
    assert all(isinstance(text,str) for text in texts)

def test_missing_semantic_columns():
    df=pd.DataFrame(
        {
            "title":["Ethiopia Coffee"],
        }
    )
    with pytest.raises(ValueError):
        build_text_embedding(df)
def test_semantic_columns():
    expected_columns={
    "title",
    "description",
    "category",
    "subcategory",
    "tags",
    "language",
    "country",
    "creator_name",
    }
    assert set(semantic_columns)==expected_columns
