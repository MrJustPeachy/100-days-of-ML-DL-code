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

# Runs all of the above functions for an easy process of testing ML algorithms
def testClassifier(classifier, params, filename):

    # Initialize the classifier and use grid search CV
    def initializeCVClassifier(classifier, params):
        return GridSearchCV(classifier, params)

    # Fits the classifier to the training data and returns the classifier
    def FitClassifier(classifier, X_train, y_train):
        return classifier.fit(X_train, y_train)

    # Predicts the probability of the result given the test data and 
    # returns the numpy array of it with the second column dropped
    def ScoreClassifiers(classifier, X_test):
        scores  = np.array(classifier.predict_proba(X_test))
        scores = np.delete(scores, 1, 1)
        return scores.flatten()

    # Creates a pandas dataframe to export the data as a CSV file
    def CreateCSV(data, filename):
        results = pd.DataFrame()
        results[''] = testing_set["ID"]
        results["Made Donation in March 2007"] = data

        # Export to CSV
        results.to_csv(filename, index=False)

    initialClassifier = initializeCVClassifier(classifier, params)
    classifierFit = FitClassifier(initialClassifier, X_train, y_train)
    classifierScores = ScoreClassifiers(classifierFit, X_test)
    CreateCSV(classifierScores, filename)

# Initialize the classifiers
LRParams = {'C': [1.0, 0.5, 0.1, 1.5, 2.0], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'max_iter': [50, 100, 150, 200]}
testClassifier(LogisticRegression(), LRParams, 'LogisticRegressionResults.csv')