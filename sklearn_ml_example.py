# This is a tutorial taken from 
# https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
# This is just simply that tutorial with some of my own notes, for my own
# edification

import os
import numpy as np
import pandas as pd

# SciKit Learn basics for this tutorial
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Family of models
from sklearn.ensemble import RandomForestRegressor

# Tools for cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Metrics we can use to evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

# Means of "persisting" our model for future use
from sklearn.externals import joblib

# Joblib is an alternative to Pickle Package, more efficient at
# storing numpy arrays

# Now it's time to load the red wine data

dataset_url = ('http://mlr.cs.umass.edu/ml/machine-learning-databases/'
               + 'wine-quality/winequality-red.csv')
data = pd.read_csv(dataset_url, sep=';')

y = data.quality
X = data.drop('quality', axis=1)

# Set aside 20% of the data as a test set for evaluating the model.
# The arbitrary random_state is a seed that lets us reproduce our result
# Stratify the sample by the target variable (y) to make evaluation metrics
# more reliable
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2, 
                                                    random_state=123,
                                                    stratify=y)
                                                    
# Since the metrics we are looking at all use different scales, we need
# to standardize it.  This makes sense to me, He Who Has Seen 10000 Iterations
# of Dimensionless Navier-Stokes.  It allows us to make sure that everything 
# is considered, even if the relative quantity is small, because its impact 
# might not be (for example CO2 in the atmosphere vs N2 or O2)

# Just remember to scale the test set and training set identically

scaler = preprocessing.StandardScaler().fit(X_train)
# This scaler object has the saved means and standard devations for each
# feature in the training set.

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


pipeline = make_pipeline(preprocessing.StandardScaler(), 
                        RandomForestRegressor(n_estimators=100))

# Now it's time to declare Hyperparameters - these are the parameters that
# we tune (whereas model parameters are derived directly from the data)

# For RandomForestRegressors, the hyperparameters are (Mean Squared Error OR
# Mean Absolute Error) and (Number of Trees)


# Declare them

hyperparameters = {'randomforestregressor__max_features' : 
                   ['auto','sqrt','log2'],
                   'randomforestregressor__max_depth' : [None, 5, 3, 1]}
                   
# Cross-validation is critical - it helps maximize model performance 
# while reducing the chance of overfitting the data

# Split your data into k equal parts, or "folds" (typically k=10).
# Preprocess k-1 training folds.
# Train your model on the same k-1 folds.
# Preprocess the hold-out fold using the same transformations from step (2).
# Evaluate your model on the same hold-out fold.
# Perform steps (2) - (5) k times, each time holding out a different fold.
# Aggregate the performance across all k folds. This is your performance metric.

# this is really easy using GridSearchCV

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train,y_train)

print(clf.best_params_)

# Train the hyperparameters using cross validation as above, and then
# refit the model with the best set of hyperparameters using the entire
# training set.  This functionality is ON by default.

# the clf object used to tune the hyperparameters can also be used directly
# like a model object.

y_predict = clf.predict(X_test)

# Evaluating model performance:
print(y_predict)
print(r2_score(y_test,y_predict))
print(mean_squared_error(y_test, y_predict))



joblib.dump(clf,'rf_regressor.pkl')