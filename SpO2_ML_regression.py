#!/usr/bin/env python3

### Objective: To build a model and predict the output of SpO2 value
### Regressors: Linear Regression, Elastic Net, K-Nearest Neighbors, Support Vector Regressor, Decision Tree, Random Forest, and Gradient Boosting Regressor 

### Load packages 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_scor

### Load hypoxemia dataset imputed, feature scaled
dataset = pd.read_csv('hypoxemia.csv', index_col=0)
# remove the binary outcome column
dataset = dataset.drop('hypoxemia_90', axis=1)
dataset.shape #(7389, 49)

### Create arrays for features and outcome variable
y = dataset['vs_spo2_value'].values
X = dataset.drop('vs_spo2_value', axis=1).values

### Features
features = dataset.columns

### Create training and test sets: 60% training - 40% test, random_state of 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)



### LINEAR REGRESSION - Ordinary Least Squares

## Untuned LR model
# Create a linear model regressor: lr
lr = LinearRegression()
# Fit the regressor to the training data
lr.fit(X_train,y_train)
# Predict on the test data: y_pred
y_pred = lr.predict(X_test)

# Return the coefficients: Residual sums of squares
print("Coefficients:", lr.coef_)
# Compute R^2 (coefficient of determination): explained variance score 
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2396
# Compute mean absolute error (MAE)
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3754
# Compute mean squared error (MSE)
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.6007
# Compute root mean square error (RMSE)
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.5497



### ELASTIC NET
## Linear regression with combined L1 and L2 priors as regularizer

## Untuned Elastic net model
# Create an elastic net regressor: net
net = ElasticNet()
# Fit the regressor to the training data
net.fit(X_train,y_train)
# Predict on the test data: y_pred
y_pred = net.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.1908
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.5209
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 13.4087
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.6618

## Specify the hyperparameter space
alpha = np.linspace(0.01, 1, 10)
l1_ratio = np.linspace(0.01, 1, 10)
max_iter = [1000, 5000]
param_grid = {'alpha':alpha,
              'l1_ratio':l1_ratio,
              'max_iter': max_iter}

# Instantiate the Elastic Net regressor
net = ElasticNet()
# Create the GridSearchCV object: gm_cv
net_cv = GridSearchCV(net, param_grid, cv=10)
# Fit it to the training data
net_cv.fit(X_train, y_train)

# Predict on the test set 
y_pred = net_cv.predict(X_test)

# Return tuned hyperparameters
print("Tuned ElasticNet parameters: {}".format(net_cv.best_params_))
# Tuned ElasticNet parameters: {'alpha': 0.12, 'l1_ratio': 0.34, 'max_iter': 1000}

## Compute metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2706
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3632
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.0866
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.4766


### K-NEAREST NEIGHBORS (KNN)
### The target is predicted by local interpolations of the targets associated of the nearest neighbors in the training set

## Untuned kNN regressor
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.1763
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.5199
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 13.6490
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # 3.6945

## Specify the hyperparameter space
n_neighbors = list(range(1,31))
weights = ["uniform", "distance"]
algorithm = ["ball_tree", "kd_tree", "brute"]
leaf_size = np.linspace(10, 100, 10)
p = np.linspace(1, 10, 10)

param_grid = {'n_neighbors': n_neighbors,
              'weights': weights,
              'algorithm': algorithm
#              'leaf_size': leaf_size,
#              'p': p
              }

# Instantiate the KNN regressor
knn = KNeighborsRegressor()

# Create the GridSearchCV object: knn_cv
knn_cv = GridSearchCV(knn, param_grid, cv=10)
# Fit it to the training data
knn_cv.fit(X_train, y_train)
print("Tuned kNN Parameter: {}".format(knn_cv.best_params_))
# Tuned kNN Parameter: {'algorithm': 'brute', 'n_neighbors': 30, 'weights': 'distance'}

# Predict on the test set 
y_pred = knn_cv.predict(X_test)

# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2510
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3784
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.4115
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.5230


### SUPPORT VECTOR REGRESSION (SVR)

## Untuned SVR
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.1981
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.2971
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 13.2884
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.6453

## Specify the hyperparameter space: 'step_name_ _parameter_name'
kernel = ["linear", "poly", "rbf"]
C = np.linspace(0.001, 1000, 10)
gamma = np.linspace(0.001, 1, 10)
epsilon = np.linspace(0.01, 1, 10)
param_grid = {
#              'kernel': kernel,
#              'epsilon': epsilon,
              'C':C,
              'gamma':gamma
              }

# Instantiate the SVR
svr_cv = SVR()
# Create the GridSearchCV object: svr_cv
svr_cv = GridSearchCV(svr_cv, param_grid, cv=10)
# Fit it to the training data
svr_cv.fit(X_train, y_train)
print("Tuned SVR Parameter: {}".format(svr_cv.best_params_))
# Tuned SVR Parameter: {'C': 111.11200000000001, 'gamma': 0.001}

# Predict on the test set 
y_pred = svr_cv.predict(X_test)

# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2496
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.2764
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.4355
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.5264



### DECISION TREE REGRESSION

## Untuned Decision tree regressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: -0.4941
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # 3.3051
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 24.7578
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 4.9757

## Setup the parameters and distributions
criterion = ["mse", "friedman_mse" , "mae"]
splitter = ["best", "random"]
max_features = randint(1,9)
max_depth = np.linspace(1,20, 10)

param_dist = {
    "criterion": criterion,
    "splitter": splitter,
    "max_depth": max_depth,
    "max_features": max_features
}

# Instantiate a Decision Tree regressor
tree = DecisionTreeRegressor()
# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=10)
# Fit to training data
tree_cv.fit(X_train, y_train)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
# Tuned Decision Tree Parameters: {'criterion': 'mse', 'max_depth': 7.333333333333334, 'max_features': 7, 'splitter': 'best'}

# Predict on the test set 
y_pred = tree_cv.predict(X_test)

print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.1747
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.4752
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 13.6752
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.6980



### RANDOM FOREST REGRESSION

## Untuned RF regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.1564
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.5822
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 13.9797
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.7389

## Setup the parameters and distributions to sample from: param_dist
n_estimators = [2000, 3000]
max_depth = np.linspace(10,20, 10)
max_features = randint(1,9)
criterion = ["mse", "mae"]

param_dist = {
              "n_estimators": n_estimators,
              "max_depth": max_depth,
              "max_features": max_features,
               "criterion": criterion
}


# Instantiate a Random Forest regressor
rf = RandomForestRegressor()
# Instantiate the RandomizedSearchCV object: tree_cv
rf_cv = RandomizedSearchCV(rf, param_dist, cv=10)

# Fit to training data
rf_cv.fit(X_train, y_train)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(rf_cv.best_params_))
# Tuned Decision Tree Parameters: {'criterion': 'mse', 'max_depth': 11.11, 'n_estimators': 3000}

# Predict on the test set 
y_pred = rf_cv.predict(X_test)

print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2512
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3933
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.4081
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.5225

## Feature importance
rfc = RandomForestRegressor(criterion ='mse', max_depth = 11.11, n_estimators = 3000)
rfc.fit(X, y)
rfc.feature_importances_




### GRADIENT BOOSTING REGRESSION

## Untuned Gradient Boosting regressor
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
# Compute model metrics
print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2666
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3751
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: 12.1526
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.4861


### Setup the parameters and distributions to sample from: param_dist
loss = ["lad"]
learning_rate = np.linspace(0.001, 1, 10)
n_estimators = [2000]
min_samples_split = randint(1, 9)
max_depth = np.linspace(10, 20, 10)
criterion = ["friedman_mse"]

param_dist = {
              "loss": loss,
              "learning_rate" : learning_rate,
              "n_estimators": n_estimators,
              "max_depth": max_depth,
              "min_samples_split": min_samples_split,
              "criterion": criterion
}


# Instantiate a Gradient Boosting regressor
gb = GradientBoostingRegressor()
# Instantiate the RandomizedSearchCV object: tree_cv
gb_cv = RandomizedSearchCV(gb, param_dist, cv=10)


# Fit to training data
gb_cv.fit(X_train, y_train)
# Print the tuned parameters and score
print("Tuned Gradient Boosting Parameters: {}".format(gb_cv.best_params_))
# Tuned Gradient Boosting Parameters: {'criterion': 'friedman_mse', 'learning_rate': 0.001, 
# 'loss': 'lad', 'max_depth': 18.88888888888889, 'min_samples_split': 8, 'n_estimators': 2000}

# Predict on the test set 
y_pred = gb_cv.predict(X_test)

print("R^2: {:.4f}".format(r2_score(y_test, y_pred))) # R^2: 0.2347
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred))) # MAE: 2.3422
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred))) # MSE: MSE: 12.6811
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred)))) # RMSE: 3.5611
