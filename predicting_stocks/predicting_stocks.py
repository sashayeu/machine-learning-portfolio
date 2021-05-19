import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

# reading the data 
# data = pd.read_csv('tesla_stock.csv')
# data = data.drop(['Date'], axis=1)

# ### FUNCTION

def close_price_prediction(data, future_days):
    data['Prediction'] = data[['Close']].shift(-future_days)

    X = np.array(data.drop(['Prediction'], axis = 1))[:-future_days]
    y = np.array(data['Prediction'])[:-future_days]

    #Creating the decision tree regressor model
    tree = DecisionTreeRegressor(max_depth = 8)

    #fitting the decision tree to the data
    tree.fit(X, y)

    #Get the observations that need to be predicted 
    # I do this by dropping the 'Prediction' column of the data and only taking the last observations, the number of 
    # which equals future_days
    x_future = data.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days) 

    #Converting this to a numpy array
    x_future = np.array(x_future)

    tree_prediction = tree.predict(x_future)

    #returns tree_prediction

    return tree_prediction

# print(type(close_price_prediction(data,100)))







