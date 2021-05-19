import pytest
import pandas as pd
import numpy as np
import coldKmeans as cKm

data = pd.read_csv('country_data.csv', sep= ",")

data_subset = data[['exports','income','gdpp']]
data_subset = data_subset.dropna()
data_subset_NP = data_subset.to_numpy()


#normalizing variables in data_subset_NP and creating subsets

exports = data_subset_NP[:,0]
mx = np.max(exports)
mn = np.min(exports)

exports_norm = (exports - mn)/(mx - mn)
exports_norm = np.around(exports_norm, decimals = 2) 

income = data_subset_NP[:,1]
mx = np.max(income)
mn = np.min(income)

income_norm = (income - mn)/(mx - mn)
income_norm = np.around(income_norm, decimals = 2) 

gdpp = data_subset_NP[:,2]
mx = np.max(gdpp)
mn = np.min(gdpp)

gdpp_norm = (gdpp - mn)/(mx - mn)
gdpp_norm = np.around(gdpp_norm, decimals = 2) 

# these are all variables against each other
exports_vs_income_vs_gdpp = np.stack((exports_norm, income_norm,gdpp_norm),axis=-1)


## Testing cold_kmeans()

def test_cold_kmeans_type():
    global exports_vs_income_vs_gdpp
    assert isinstance(cKm.cold_kmeans(exports_vs_income_vs_gdpp, 6, 2019), tuple)

def test_cold_kmeans_shape():
	expected = 2
	assert len(cKm.cold_kmeans(exports_vs_income_vs_gdpp, 6, 2019)) == expected

## Testing looping_kmeans()

def test_looping_kmeans_type():
	assert isinstance(cKm.looping_kmeans(exports_vs_income_vs_gdpp,
		list(range(1,15))), list)

def test_looping_kmeans_size():
	expected = 5
	assert len(cKm.looping_kmeans(exports_vs_income_vs_gdpp,
		list(range(1,6)))) == expected

def test_looping_kmeans_goodness():
	out = cKm.looping_kmeans(exports_vs_income_vs_gdpp,list(range(1,6)))
	assert (out[1:] <= out[:-1])
