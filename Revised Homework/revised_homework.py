from scipy.spatial import distance  
import pandas as pd
import numpy as np


# My cold kmeans implementation, which takes in the array, the number of clusters, and a random state
# and outputs the cluster centers, as well as the cluster to which each point belongs to

def cold_kmeans(arrayName, k, randomState):
    df = pd.DataFrame(arrayName)
    
    #finding k centers 
    centers= df.sample(k, random_state = randomState)
    centers_np = centers.to_numpy()

    #oldCenters, used for storing the centers of the current iteration, is initialized
    oldCenters = []

    # The centers are recalculated until the centers from the new calculations are equal to the centers 
    # from the previous calculation. Therefore, the stopping condition is when the clusters stop 
    # changing and all centers remain the same. 
    
    # To avoid falling into a local minima, a second stopping condition is used: number of iterations
    i = 0
    
    while (not (np.array_equal(oldCenters, centers_np)) and i<300):
        #iteration number increased
        i+=1
        
        #old Centers are set to whatever centers were calculated before 
        oldCenters = np.copy(centers_np)
        
        #new distances calculated and clusters are assigned
        dists = distance.cdist(arrayName, centers_np, 'euclidean')
        clusters = np.argmin(dists, axis=1)

        #looping over each cluster
        for n in range (len(centers_np)):
            subset = []
            
            #adding the indices of the points in each cluster
            for i in range (len(clusters)):
                if (clusters[i]==n):
                    subset.append(i)
                    
            #taking a subset of the points in the cluster
            sub = df.iloc[subset]
            
            #finding a new center for this cluster and changing its value in centers_np
            centers_np[n] = sub.mean(axis=0)
    
    return centers_np, clusters


# This function repeatedly runs my kmeans algorithm with different values of k to determine which number
# of clusters is ideal.

def looping_kmeans(arrayName,kList):

    output = []

    for k in kList:
        within_cluster_sumsqs = 0
        
        output_of_kmeans = cold_kmeans(arrayName, k, 10)
        
        centers = output_of_kmeans[0]

        for c in range(0,k):
            # Extract the cluster's center and associated points:
            cluster_center = [centers[c,:]]
            cluster_points = arrayName[np.where(output_of_kmeans[1] == c)]
            
            # Compute the following for each cluster:
            cluster_spread = distance.cdist(cluster_points, cluster_center, 'euclidean')
            cluster_total = np.sum(cluster_spread)
            
            # Add this cluster's within sum of squares to within_cluster_sumsqs
            within_cluster_sumsqs = within_cluster_sumsqs + cluster_total
        
        output.append(within_cluster_sumsqs)
    
    return output