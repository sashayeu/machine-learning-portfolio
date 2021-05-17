# Cold Kmeans 

Over the course of the semester, we learned about many unsupervised algorithms. My favourite one by far was K-means, and in this jupyter notebook I will be showing my cold-implementation skills of K-means. Cold-implementations are algorithms coded from scratch, using basic elements of the language and few packages to help. To determine the best value of k, I use a technique called elbowology. Although elbowology still leaves room for human interpretation, this is a good way to visualize the effect of different number of groups. To wrap up this element, I will be comparing my implementation of K-means to the existing off-the-shelf sklearn version, as well as discussing the limitations of K-means. 

To guide you through this, I will be working with a dataset about the countries of the world. I will be exploring how these countries can be grouped together based on exports, income, and GDP per capita. 

My cold implementation of K-means, as well as the looping_kmeans helped function, are located in the coldKmeans.py file. Unit tests for these functions are also included. 