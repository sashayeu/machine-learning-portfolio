# Predicting Heart Attack victims

Over the course of the semester, we learned about many supervised algorithms. So many, that it may seem difficult to choose the correct algorithm to solve a problem. That's why in this notebook, I will be showing my skills of choosing the best model with the best parameters. Parameters of models are human provided inputs, such as how many neighbours should kNN have, or what is the learning step in gradient descent. My goal in this element is to reason through two possible models to pick the better one. 

There are many different ways to assess models. In this element, I will be using mean squared error, along with cross validation. Mean squared error is a simple yet powerful concept. It squares the error between each predicted value and the true value, then takes the mean for all the predicted values. I like this way of measuring success because it is easy to implement and intuitive. With only two inputs, the predicted vector and the truth vector, I created a function for it in my jupyter notebook. 

However, computing the mean squared error just once for an algorithm working on a set of data is risky. What if that specific truth vector had a lot of outliers, so a high mean squared error was bound to happen even if the model was accurate? This is why I also implemented 5 fold cross validation. This technique divides data up into 5 chunks, then uses each chunk as the testing data and the other 4 as the training data. Running the algorithm 5 times, each time on a different chunk being the testing data, gives you a mean of the mean squared errors, which is a more accurate predictor of the general accuracy of the model.

With these two tools, I chose parameters that I thought fit my data best. I also included the 5 fold cross validation code in a separate file to go along with unit tests.

In the end, to choose the best model from the two I created, I used a combination of mean squared error, as well as timing how long each one took. I specifically used the timeit module because of its precision, as well as because it runs the algorithm multiple times to get the mean time. It's important to recognize the strengths and fallbacks for each method, but in the end one must outweigh the other. 

Open the supervised_algorithms.ipynb file to view more!