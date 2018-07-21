# Start out with standard ML models initially
# Then use tensorflow/keras to create a deep learning implementation of this problem

# This is a classification problem

from sklearn.linear_model import LogisticRegression
    # LogisticRegression is a linear classifier that uses
    # error minimizing formulas to come to the line of best fit
    # http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

from sklearn.naive_bayes import GaussianNB
    # Combines Gaussian and Naive Bayes in order to create classification for probablity
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

# Other modules that are needed
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Read in the data into a pandas dataframe
training_set = pd.read_csv('training.csv')
X_test = pd.read_csv('test.csv')

# Get the list of columns and set the first column to the ID
training_columns = training_set.columns.tolist()
training_columns[0] = "ID"
training_set.columns = training_columns

testing_columns = X_test.columns.tolist()
testing_columns[0] = "ID"
X_test.columns = testing_columns

# Save the results of the training data in a seperate var
y_train = np.array(training_set[['Made Donation in March 2007']])

# Remove the results from the training set
X_train = np.array(training_set.drop(['Made Donation in March 2007'], axis=1))

# Initialize the classifiers
LRClassifier = LogisticRegression()

# Set the list of the parameters that we want to test
LRParams = {'C': [1.0, 0.5, 0.1, 1.5, 2.0], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'max_iter': [50, 100, 150, 200]}

# Initialize the classifier as a GridSearchCV object
LRCV = GridSearchCV(LRClassifier, LRParams)

# Fit the classifiers
LRCV.fit(X_train, y_train)

# See the initial score of the classifiers with no hyperparameters tuned
lrData = np.array(LRCV.predict_proba(X_test))
lrData = np.delete(lrData, 1, 1) # THANK YOU @ https://stackoverflow.com/questions/1642730/how-to-delete-columns-in-numpy-array
print(lrData)

# Set up the data results in pandas dataframe
lrResults = pd.DataFrame()

# set the first row to the ideas used in testing
lrResults[''] = X_test["ID"]
# Set the predictions using the functions from before
lrResults["Made Donation in March 2007"] = lrData.flatten()
# Export file to a CSV
lrResults.to_csv('LogisticRegressionResults.csv', index=False)