import pytest
import numpy as np
import pandas as pd

import supervised_algorithms_code

#importing the data file
data_pd = pd.read_csv("heart.csv", sep = ",")

#data mannipulation
data_pd = data_pd.drop(labels = ['sex','cp','fbs','restecg','exng','oldpeak','slp','thall','caa'], axis = 1)
data_pd = data_pd.dropna(axis=0)
df_random = data_pd.sample(frac=1,random_state=15)

#data_np created to use for models and dividing data
data_np = df_random.to_numpy()


def test_compute_mse_type():
    truth_vec = np.array([0,0,0])
    predict_vec = np.array([0,0,1])

    out = supervised_algorithms_code.compute_mse(truth_vec, predict_vec)
    assert isinstance(out, float) 

def test_compute_mse_value():
    truth_vec = np.array([0,0,0])
    predict_vec = np.array([0,0,1])

    out = supervised_algorithms_code.compute_mse(truth_vec, predict_vec)

    assert out == 1/3

def test_kNN_CV_type():
    out = supervised_algorithms_code.kNN_CV(data_np,1)

    assert isinstance(out, float)

def test_kNN_CV_error():
    with pytest.raises(ValueError) as e_info:
        supervised_algorithms_code.kNN_CV(data_np,0)

def test_dt_CV_type():
    out = supervised_algorithms_code.dt_CV(data_np,1)

    assert isinstance(out, float) 

def test_dt_CV_error():
    with pytest.raises(ValueError) as e_info:
        supervised_algorithms_code.dt_CV(data_np,0)


    
