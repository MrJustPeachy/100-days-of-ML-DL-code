# Start out with standard ML models initially
# Then use tensorflow/keras to create a deep learning implementation of this problem

# This is a classification problem

from sklearn.linear_model import LogisticRegression
    # LogisticRegression is a linear classifier that uses
    # error minimizing formulas to come to the line of best fit
    # http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

from sklearn.linear_model import SGDClassifier
    # Stochastic Gradient Descent uses convex loss functions like SVM's and Logistic Regression.
    # This function has been around for a while, but has gained popularity due to a boost of large-scale learning / big data.
    # http://scikit-learn.org/stable/modules/sgd.html

from sklearn.neighbors import KNeighborsClassifier
    # Uses the kNN algorithm as a classifier. 
    # kNN is more commonly used in unsuperivsed learning
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
    # Uses the Gaussian processes to create probabilistic classification
    # http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB
    # Combines Gaussian and Naive Bayes in order to create classification for probablity
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

from sklearn.tree import DecisionTreeClassifier
    # Classifier used to perform multi-class classification.
    # Decision Trees can increase in depth and complexity, which can lead to either overfitting or higher accuracy.
    # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# Other modules that are needed
import numpy as np
import pandas as pd

# Read in the data into a pandas dataframe
training_set = pd.read_csv('training.csv')
testing_set = pd.read_csv('test.csv')

# Get the list of columns and set the first column to the ID
columns = training_set.columns.tolist()
columns[0] = "ID"

# Save the results of the training data in a seperate var
y_train = np.array(training_set[['Made Donation in March 2007']])
y_test = np.array(testing_set[['Made Donation in March 2007']])

# Remove the results from the training set
X_train = np.array(training_set.drop(['Made Donation in March 2007'], axis=1))
X_test = np.array(testing_set.drop(['Made Donation in March 2007'], axis=1))


# Initialize the classifiers
LRClassifier = LogisticRegression()
StochasticGDClassifier = SGDClassifier()
KNNClassifier = KNeighborsClassifier()
GPClassifier = GaussianProcessClassifier()
GNPClassifier = GaussianNB()
DTClassifier = DecisionTreeClassifier()

# Fit the classifiers
LRClassifier.fit(X_train, y_train)
StochasticGDClassifier.fit(X_train, y_train)
KNNClassifier.fit(X_train, y_train)
GPClassifier.fit(X_train, y_train)
GNPClassifier.fit(X_train, y_train)
DTClassifier.fit(X_train, y_test)

# See the initial score of the classifiers with no hyperparameters tuned
print(LRClassifier.score(X_test, y_test))