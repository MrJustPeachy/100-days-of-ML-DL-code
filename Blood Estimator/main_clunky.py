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
training_columns = training_set.columns.tolist()
training_columns[0] = "ID"
training_set.columns = training_columns

testing_columns = testing_set.columns.tolist()
testing_columns[0] = "ID"
testing_set.columns = testing_columns

# Save the results of the training data in a seperate var
y_train = np.array(training_set[['Made Donation in March 2007']])

# Remove the results from the training set
X_train = np.array(training_set.drop(['Made Donation in March 2007'], axis=1))
X_test = testing_set


# Initialize the classifiers
LRClassifier = LogisticRegression()
StochasticGDClassifier = SGDClassifier(loss="log")
KNNClassifier = KNeighborsClassifier()
GPClassifier = GaussianProcessClassifier()
GNBClassifier = GaussianNB()
DTClassifier = DecisionTreeClassifier()

# Fit the classifiers
LRClassifier.fit(X_train, y_train)
StochasticGDClassifier.fit(X_train, y_train)
KNNClassifier.fit(X_train, y_train)
GPClassifier.fit(X_train, y_train)
GNBClassifier.fit(X_train, y_train)
DTClassifier.fit(X_train, y_train)

# See the initial score of the classifiers with no hyperparameters tuned
lrData = np.array(LRClassifier.predict_proba(X_test))
sgdcData = StochasticGDClassifier.predict_proba(X_test)
knnData = KNNClassifier.predict_proba(X_test)
gpData = GPClassifier.predict_proba(X_test)
gnbData = GNBClassifier.predict_proba(X_test)
dtData = DTClassifier.predict_proba(X_test)
lrData = np.delete(lrData, 1, 1) # THANK YOU @ https://stackoverflow.com/questions/1642730/how-to-delete-columns-in-numpy-array

# Set up the data results in pandas dataframe
lrResults = pd.DataFrame()

# set the first row to the ideas used in testing
lrResults[''] = testing_set["ID"]
# Set the predictions using the functions from before
lrResults["Made Donation in March 2007"] = lrData.flatten()
# Export file to a CSV
lrResults.to_csv('LogisticRegressionResults.csv', index=False)