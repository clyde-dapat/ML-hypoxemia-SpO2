#!/usr/bin/env python3

### Objective: To build a model and predict the output of a binary outcome if a child has hypoxemia (SpO2<90) or not
# Classifiers: Logistic Regression, K-Nearest Neighbors, Kernel SVM, Naive Bayes, Decision Trees, Random Forest, and Gradient Boosting

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
from sklearn.model_selection import cross_val_score
# upsampling for imbalanced classes
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Cohen's kappa score
from sklearn.metrics import cohen_kappa_score


### Load hypoxemia dataset 
dataset = pd.read_csv('hypoxemia.csv', index_col=0)
dataset.shape # (7389, 50)
# remove the continuous outcome column
dataset = dataset.drop('vs_spo2_value', axis=1)
dataset['hypoxemia_90'].value_counts(dropna=False)
# nonhypoxemia    6873
# hypoxemia        516

###  convert labels as binary coding 0,1
hypoxemia = pd.get_dummies(dataset['hypoxemia_90'])
hypoxemia.head()

### Create arrays for features and outcome variable
# y = hypoxemia['hypoxemia'].values
# X = dataset.drop('hypoxemia_90', axis=1).values

### Create arrays for features and outcome variable
y = hypoxemia.hypoxemia
X = dataset.drop('hypoxemia_90', axis=1)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

# Concatenate training set back together
X = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
nonhypoxemia = X[X.hypoxemia==0]
hypoxemia = X[X.hypoxemia==1]

# Upsample minority class
hypoxemia_upsampled = resample(hypoxemia,
                        replace=True,
                        n_samples=len(nonhypoxemia),
                        random_state=0)

# Combine majority and upsampled minority
upsampled = pd.concat([nonhypoxemia, hypoxemia_upsampled])

# Check new class counts
upsampled.hypoxemia.value_counts()
# 1    4123
# 0    4123
# Name: hypoxemia, dtype: int64


### Run upsampled data on best model with hypertuned parameters


### Logistic Regression
### Probabilities describing the possible outcomes are modeled using a logistic function

## Run logit model with upsampled dataset
y_train = upsampled.hypoxemia
X_train = upsampled.drop('hypoxemia', axis=1)
logreg_up = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train, y_train)
# Predict labels of the test set
y_pred = logreg_up.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# [[2083  667]
#  [  66  140]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9693    0.7575    0.8504      2750
#            1     0.1735    0.6796    0.2764       206
#    micro avg     0.7520    0.7520    0.7520      2956
#    macro avg     0.5714    0.7185    0.5634      2956
# weighted avg     0.9138    0.7520    0.8104      2956

# Check accuracy score
accuracy_score(y_test, y_pred) # 0.7520
# Check F1 score
f1_score(y_test, y_pred) # 0.2764
# Check Cohen's kappa scote
cohen_kappa_score(y_test, y_pred) # 0.1860

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg_up.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7639


# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
# Specify the hyperparameter space
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression(solver='liblinear', max_iter=1000)
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
# Fit it to the training data
logreg_cv.fit(X_train, y_train)
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
# Tuned Logistic Regression Parameters: {'C': 0.006105402296585327, 'penalty': 'l2'}
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
# Tuned Logistic Regression Accuracy: 0.9336792240018047

# Predict the labels of the test set: y_pred
y_pred = logreg_cv.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# [[2742    8]
# [ 189   17]]
print(classification_report(y_test, y_pred, digits=4))
#              precision    recall  f1-score   support

#            0     0.9355    0.9971    0.9653      2750
#            1     0.6800    0.0825    0.1472       206

#    micro avg     0.9334    0.9334    0.9334      2956
#    macro avg     0.8078    0.5398    0.5563      2956
# weighted avg     0.9177    0.9334    0.9083      2956

f1_score(y_test, y_pred)
# 0.1471861471861472

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7773345101500442

## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Reg')
plt.show()



### K-NEAREST NEIGHBORS
### Classifier computes from a simple majority vote of the nearest neighbors of each point, 
### where a query point is assigned the data class which has the most representatives within the nearest neighbors of the point


# Create the hyperparameter grid
c_space = list(range(1,31))
# Specify the hyperparameter space
param_grid = {'n_neighbors': c_space}
# Instantiate a kNN classifier: knn
knn = KNeighborsClassifier()
# Instantiate the GridSearch CV: knn_cv
knn_cv = GridSearchCV(knn, param_grid, cv=10)
# Fit to training data
knn_cv.fit(X_train, y_train)
print("Tuned kNN Parameter: {}".format(knn_cv.best_params_))
# Tuned kNN Parameter: {'n_neighbors': 21}
print("Tuned KNN Accuracy: {}".format(knn_cv.best_score_))
# Tuned KNN Accuracy: 0.9334536431310625
## Predict the labels of the test set: y_pred
y_pred = knn_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2741    9]
#  [ 185   21]]

print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support
#            0     0.9368    0.9967    0.9658      2750
#            1     0.7000    0.1019    0.1780       206
#    micro avg     0.9344    0.9344    0.9344      2956
#    macro avg     0.8184    0.5493    0.5719      2956
# weighted avg     0.9203    0.9344    0.9109      2956

f1_score(y_test, y_pred)
# 0.17796610169491528

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = knn_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7374342453662841

## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - kNN')
plt.show()



### SUPPORT VECTOR MACHINE (SVM)
### SVM classifier fits the widest possible margins (decision boundary) between classes

# Specify the hyperparameter space: 'step_name_ _parameter_name'
c_space = np.linspace(0.001, 1000, 10)
gamma_space = np.linspace(0.001, 1, 10)
param_grid = {'C':c_space,
              'gamma':gamma_space}
# Instantiate a C-Support Vector Classifier (SVC) with Gaussian RBF Kernel
svm = SVC(probability=True)
# Instantiate the GridSearch CV: svm_cv
svm_cv = GridSearchCV(svm, param_grid, cv=10)
# Fit to training data
svm_cv.fit(X_train, y_train)
print("Tuned SVM Parameters: {}".format(svm_cv.best_params_))
# Tuned SVM Parameters: {'C': 111.11200000000001, 'gamma': 0.001}

print("Tuned SVM Accuracy: {}".format(svm_cv.best_score_))
# Tuned SVM Accuracy: 0.9323257387773517

## Predict the labels of the test set: y_pred
y_pred = svm_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2735   15]
#  [ 188   18]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9357    0.9945    0.9642      2750
#            1     0.5455    0.0874    0.1506       206

#    micro avg     0.9313    0.9313    0.9313      2956
#    macro avg     0.7406    0.5410    0.5574      2956
# weighted avg     0.9085    0.9313    0.9075      2956

f1_score(y_test, y_pred)
# 0.1506276150627615

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = svm_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.6819752868490733

## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.show()


### NAIVE BAYES CLASSIFIER
### Utilizes Bayes' theorem with the "naive" assumption of conditional independence between every pair of 
### features given the value of class variable

# Specify the hyperparameter space: 'param_grid'
var_space = np.linspace(0.01,10,10)
param_grid = {"var_smoothing": var_space}
# Instantiate a Gaussian Naive Bayes classifier: gnb
gnb = GaussianNB()
# Instantiate the RandomizedSearchCV object: tree_cv
gnb_cv = GridSearchCV(gnb, param_grid, cv=10)
# Fit to training data
gnb_cv.fit(X_train,y_train)
print("Tuned GNB Parameter: {}".format(gnb_cv.best_params_))
# Tuned GNB Parameter: {'var_smoothing': 4.45}
print("Tuned GNB Accuracy: {}".format(gnb_cv.best_score_))
# Tuned GNB Accuracy: 0.9307466726821566

## Predict the labels of the test set: y_pred
y_pred = knn_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2741    9]
#  [ 185   21]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9368    0.9967    0.9658      2750
#            1     0.7000    0.1019    0.1780       206

#    micro avg     0.9344    0.9344    0.9344      2956
#    macro avg     0.8184    0.5493    0.5719      2956
# weighted avg     0.9203    0.9344    0.9109      2956

f1_score(y_test, y_pred)
# 0.17796610169491528

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = gnb_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7595339805825243
## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - NB')
plt.show()



### DECISION TREE CLASSIFIER
### Non-parametric supervised learning method that creates a model that predicts the value of a target variable
### by learning simple decision rules inferred from the data features

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=10)
# Fit to training data
tree_cv.fit(X_train,y_train)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
# Tuned Decision Tree Parameters: {'criterion': 'gini', 'max_depth': 3, 'max_features': 4, 'min_samples_leaf': 3}
print("Tuned Decision Tree Accuracy: {}".format(tree_cv.best_score_))
# Tuned Decision Tree Accuracy: 0.9318745770358674

## Predict the labels of the test set: y_pred
y_pred = tree_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2744    6]
#  [ 204    2]]

print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9308    0.9978    0.9631      2750
#            1     0.2500    0.0097    0.0187       206

#    micro avg     0.9290    0.9290    0.9290      2956
#    macro avg     0.5904    0.5038    0.4909      2956
# weighted avg     0.8834    0.9290    0.8973      2956

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = tree_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7049055604589585
## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.show()



### RANDOM FOREST CLASSIFIER
### An ensemble classifier that combines several randomized decision trees
### Each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set

## Untuned classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# [[2730   20]
#  [ 188   18]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9363    0.9945    0.9646      2750
#            1     0.5714    0.0971    0.1660       206

#    micro avg     0.9320    0.9320    0.9320      2956
#    macro avg     0.7539    0.5458    0.5653      2956
# weighted avg     0.9109    0.9320    0.9089      2956

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = rf.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7224880847308033

# Return the feature importance: higher values are more important
rf.feature_importances_


### Setup the parameters and distributions to sample from: param_dist
n_estimators = [2000, 3000]
max_depth = np.linspace(10,20, 10, endpoint=True)
min_samples_split = np.linspace(0.1, 1, 10, endpoint=True)
min_samples_leaf = randint(1, 9)
max_features = randint(1, 9)
criterion = ["gini", "entropy"]
param_dist = {"n_estimators": n_estimators,
              "max_depth": max_depth,
#              "min_samples_split":min_samples_split,
#              "max_features": max_features,
#              "min_samples_leaf": min_samples_leaf,
               "criterion": criterion}


# 0.9341 # {'n_estimators': 3000, 'max_depth': 11.11111111111111, 'criterion': 'gini'}
# 0.9341 # {'n_estimators': 2000, 'criterion': 'gini'}
# 0.9337 # {'n_estimators': 2000 }
# 0.9337 # {'n_estimators': 5000, 'criterion': 'gini'}
# 0.9301 # {'n_estimators': 3000, 'min_samples_split': 0.2, 'max_depth': 20.0, 'criterion': 'entropy'}
# 0.9301 # {'criterion': 'gini', 'max_depth': 11.11111111111111, 'max_features': 6, 'min_samples_split': 0.1, 'n_estimators': 3000}
# 0.9301 # {'criterion': 'gini', 'max_depth': 16.666666666666668, 'max_features': 5, 'min_samples_leaf': 4, 'min_samples_split': 0.2, 'n_estimators': 2000}

# Instantiate a Random Forest classifier: rf
rf = RandomForestClassifier()
# Instantiate the RandomizedSearchCV object: rf_cv
rf_cv = RandomizedSearchCV(rf, param_dist, cv=10)
# Fit to training data
rf_cv.fit(X_train,y_train)
# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))
# Tuned Random Forest Parameters: {'n_estimators': 3000, 'max_depth': 11.11111111111111, 'criterion': 'gini'}
print("Tuned Random Forest Accuracy: {:.4f}".format(rf_cv.best_score_))
# Tuned Random Forest Accuracy: 0.9341

y_pred = rf_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2747    3]
# [ 194   12]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9340    0.9989    0.9654      2750
#            1     0.8000    0.0583    0.1086       206

#    micro avg     0.9334    0.9334    0.9334      2956
#    macro avg     0.8670    0.5286    0.5370      2956
# weighted avg     0.9247    0.9334    0.9057      2956

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = rf_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.8197
## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.show()

## Feature importance
rfc = RandomForestClassifier(n_estimators=3000, max_depth=11.11, criterion='gini')
rfc.fit(X_train,y_train)
rfc.feature_importances_



### GRADIENT BOOSTING CLASSIFIER
### Gradient Tree Boosting is a generalization of boosting to arbitrary differentiable loss functions
### sequentially adds predictors to an ensemble, each one correcting its predecessor

## Untuned classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9366    0.9931    0.9640      2750
#            1     0.5250    0.1019    0.1707       206

#    micro avg     0.9310    0.9310    0.9310      2956
#    macro avg     0.7308    0.5475    0.5674      2956
# weighted avg     0.9079    0.9310    0.9087      2956
print(confusion_matrix(y_test, y_pred))
# [[2731   19]
#  [ 185   21]]
y_pred_prob = gb.predict_proba(X_test)[:,1]
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7959

### Setup the parameters and distributions to sample from: param_dist
loss = ["deviance", "exponential"]
learning_rate = np.linspace(0.001, 1, 10, endpoint=True)
n_estimators = [100, 1000, 2000]
min_samples_split = randint(1, 9)
max_depth = np.linspace(10, 20, 10, endpoint=True)

param_dist = {"loss": loss,
              "learning_rate" : learning_rate,
              "n_estimators": n_estimators,
              "max_depth": max_depth,
              "min_samples_split": min_samples_split}

# Instantiate a Gradient Boosting classifier: rf
gb = GradientBoostingClassifier()
# Instantiate the RandomizedSearchCV object: rf_cv
gb_cv = RandomizedSearchCV(gb, param_dist, cv=10)
# Fit to training data
gb_cv.fit(X_train,y_train)
# Print the tuned parameters and score
print("Tuned Gradient Boosting Parameters: {}".format(gb_cv.best_params_))
# {'learning_rate': 0.223, 'loss': 'exponential', 'max_depth': 16.666666666666668, 'min_samples_split': 7, 'n_estimators': 2000}
print("Tuned Gradient Boosting Accuracy: {:.4f}".format(gb_cv.best_score_))
# 0.9276

y_pred = gb_cv.predict(X_test)
# Compute the confusion matrix
print(confusion_matrix(y_test, y_pred))
# [[2719   31]
#  [ 178   28]]
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9386    0.9887    0.9630      2750
#            1     0.4746    0.1359    0.2113       206

#    micro avg     0.9293    0.9293    0.9293      2956
#    macro avg     0.7066    0.5623    0.5872      2956
# weighted avg     0.9062    0.9293    0.9106      2956

## AUC computation and plotting
## Compute predicted probabilities: y_pred_prob
y_pred_prob = gb_cv.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))
# AUC: 0.7908
## Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.show()
