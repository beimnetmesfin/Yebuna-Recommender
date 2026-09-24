import pandas as pd
from app.business_rule import BusinessRules

def test_filter_active():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "is_active":[True,False]
        }
    )
    rules=BusinessRules()
    result=rules.filter_active(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_visible():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "is_visible":[True,False]
        }
    )
    rules=BusinessRules()
    result=rules.filter_visible(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_approved():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "approval_status":["approved","pending"]
        }
    )
    rules=BusinessRules()
    result=rules.filter_approved(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_stock_in():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "stock":[10,0]
        }
    )
    rules=BusinessRules()
    result=rules.filter_in_stock(candidates)
    assert result["item_id"].tolist()==[1]
