# Using machine learning in predicting hypoxemia in children with respiratory virus infections

## Motivation

**Background**: Treatment guidelines for hospitalized children with pneumonia include oxygen administration based on monitoring of arterial blood oxygenation using pulse oximeter. However, in developing countries, the availability of such equipment is limited. Thus, identifying respiratory signs and other clinical factors that can predict hypoxemia would be useful in deciding whether a child requires oxygen therapy or not.   

**Objective**: To identify a model that would classify and predict hypoxemia in children infected with respiratory viruses

**Methods**: The data was combined from hospital-based and household-based cohort studies in the Philippines, 2008-2016. SpO<sub>2</sub> measurements were taken before oxygen administration. The dataset contains more than 7,000 observations. The outcome variable is hypoxemia, which is defined as having an SpO<sub>2</sub> value <90%. The predictor variables include demographic information such as age, sex, etc; illness history such as fever, coryza, cough, etc.; vital signs such as respiratory rate, heart rate, etc.; physical examination results such as wheeze, rales, chest indrawing, etc.; and viral etiology from lab-confirmed test results for respiratory syncytial virus, rhinovirus, influenza, etc. Various machine learning algorithms were used for training and model prediction including K-Nearest Neighbors, Linear and Logistic Regression, Support Vector Machines, Naive Bayes, Decision Trees, Random Forest, and Gradient Boosting. Receiver operating characteristic curves were used to evaluate the models for classification models while root-mean squared error (RMSE) and mean absolutet error (MAE) values were used to evaluate the performance of regression models. The dataset was randomized and divided into 60% training set and 40% test set with 10 iterations for cross-validation. Grid search was performed to tune hyperparameters during cross-validation.

**Results**: Random forest was identified as the best model for hypoxemia classification with the highest area under curve (AUC) of 0.82. The linear regressor, Elastic Net, was identified as the best model for predicting SpO<sub>2</sub> value with the lowest RMSE of 3.5.

## Software dependencies
Codes were run using Python version 3.7.9. Python packages used during machine learning are listed in the scripts. 

## Codes
### *hypoxemia_ML_classification.py*
This file contains the code for hypoxemia classification.

## *SpO<sub>2</sub>_ML_regression.py*
This file contains the code for predicting SpO<sub>2</sub>.