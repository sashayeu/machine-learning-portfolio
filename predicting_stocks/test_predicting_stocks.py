import pytest
import numpy as np
import pandas as pd

import predicting_stocks

data = pd.read_csv('tesla_stock.csv')
data = data.drop(['Date'], axis=1)

def test_close_price_prediction_type():


    result = predicting_stocks.close_price_prediction(data,100)

    assert isinstance(result, np.ndarray) 

def test_close_price_prediction_length():

    future_days = 100

    result = predicting_stocks.close_price_prediction(data,future_days)

    assert len(result) == future_days

def test_close_price_prediction_shape():

    future_days = 100

    result = predicting_stocks.close_price_prediction(data,future_days)

    assert result.shape == (future_days,)