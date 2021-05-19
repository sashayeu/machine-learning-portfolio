import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.cluster import KMeans 

# a function to help compute the MSE for two vectors
def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)

#5 fold cross validation for different number of depths for decision tree
def dt_CV(data,max_depth):
    
    divisor_num = 5
    
    data_divisor = len(data)//divisor_num
    test_errors = []

    for i in range (0,divisor_num):
        #splits data into testing and training, if on 10th part of data, takes remaining section of data
        if i != divisor_num-1:
            training_data = np.delete(data,slice(i*data_divisor,i*data_divisor+data_divisor),0)
            testing_data = data[i*data_divisor:i*data_divisor+data_divisor,:]
        else:
            training_data = np.delete(data,slice(i*data_divisor-1,-1),0)
            testing_data = data[i*data_divisor-1:-1,:]
        
        #assigns in classes and out classes for kNN to be fitted to
        in_classes = training_data[:,0:4]
        out_class = training_data[:,4]
        
        #defining the algorithm
        dt = DecisionTreeClassifier(max_depth = max_depth)
        
        #fitting the algorithm to the training data
        dt.fit(in_classes, out_class)
        
        #predicting values for the testing data
        all_labels = dt.predict(testing_data[:,0:4])
        
        #computing the MSE and appending it to the list of all errors
        test_error = compute_mse(testing_data[:,4],all_labels)
        test_errors.append(test_error)
    
    #returns the average MSE for kNN with a specified number of neighbours
    return np.mean(test_errors)


#5 fold cross validation for different number of neighbors for kNN
def kNN_CV(data,n_neighbors):
    
    data_divisor = len(data)//5
    test_errors = []

    for i in range (0,5):
        #splits data into testing and training, if on 10th part of data, takes remaining section of data
        if i != 4:
            training_data = np.delete(data,slice(i*data_divisor,i*data_divisor+data_divisor),0)
            testing_data = data[i*data_divisor:i*data_divisor+data_divisor,:]
        else:
            training_data = np.delete(data,slice(i*data_divisor-1,-1),0)
            testing_data = data[i*data_divisor-1:-1,:]
        
        #assigns in classes and out classes for kNN to be fitted to
        in_classes = training_data[:,0:4]
        out_class = training_data[:,4]
        
        #defining the algorithm
        kNN_alg = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        #fitting the algorithm to the training data
        kNN_alg.fit(in_classes,out_class)
        
        #predicting values for the testing data
        all_labels = kNN_alg.predict(testing_data[:,0:4])
        
        #computing the MSE and appending it to the list of all errors
        test_error = compute_mse(testing_data[:,4],all_labels)
        test_errors.append(test_error)
    
    #returns the average MSE for kNN with a specified number of neighbours
    return np.mean(test_errors)
        