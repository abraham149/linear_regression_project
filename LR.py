"""
Author: Abraham Lemus Ruiz
Last change: 7pm 8/12/2020
linkedin: https://www.linkedin.com/in/abraham-lemus-ruiz/
"""


import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def model_metrics(y_test,y_pred):   #Calculate some metrics for the model
    mse = mean_squared_error(y_test,y_pred)
    print("Mean squared error: %.4f"
      % mse)
    #r2 = r2_score(y_test, y_pred)
    #print('R2 score: %.4f' % r2 )
    rmse = mse/2
    print('RMSE score: %.4f' % rmse )


#Since this is a regression problem, we'll use an LR predictor inside a function for easy testing the changes
#It just tests the basic LinearRegression model on demand
def test_lineal_regression(features, target, model = LinearRegression()):
    #Divide the data set for testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state = 133)

    #Create and fit basic LR model
    model = model.fit(X_train, y_train)
    scores=cross_val_score(model,X_train,y_train,cv=5,n_jobs=-1)

    #Evaluate model
    y_pred = model.predict(X_test)
    #print(str(len(model.coef_))+" features: "+ str(model.coef_))
    print("Cross Val R2 Score: "+ str(scores.mean().round(4)))
    model_metrics(y_test, y_pred)

#Reading the data
print("Reading the data")
df_training_dataset = pd.read_csv(r'train_dataset_digitalhouse.csv')


## Defining pipeline from notebook
numeric_features = ["EDAD", "EXPERIENCIA", "AVG_DH", "MINUTES_DH"]
categorical_features = ["GENERO", "NV_ESTUDIO", "ESTUDIO_PREV"]
numeric_for_knnimputer1 = [ "EDAD", "EXPERIENCIA"]
numeric_for_knnimputer2 = [ "AVG_DH", "MINUTES_DH"]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer1 = Pipeline(steps=[
    ('knn',KNNImputer()),
    ('scaler', StandardScaler()),
    ('power-transform', PolynomialFeatures(degree = 3))
])

numeric_transformer2 = Pipeline(steps=[
    ('knn2',KNNImputer()),
    ('scaler2', StandardScaler()),
    ('power-transform2', PolynomialFeatures(degree = 3))
])

preprocessor = ColumnTransformer(
    [
     ('ed_exp', numeric_transformer1, numeric_for_knnimputer1),
     ('min_avg', numeric_transformer2, numeric_for_knnimputer2),
     ('cat', categorical_transformer, categorical_features)
    ],
    remainder="drop")

#model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(**Ridge_GS.best_params_))])

#Using the pipeline
print("Transforming the data with the pipeline")
X = df_training_dataset.drop(columns = ["DIAS_EMP"])
y = df_training_dataset["DIAS_EMP"]
X = preprocessor.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 133)
best_params = {'alpha': 1, 'fit_intercept': True, 'solver': 'sparse_cg', 'tol': 0.0001}

print("Training scikit Ridge model with best params")
model = Ridge(**best_params).fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
model_metrics(y_test, y_pred)


print("Training LRByHand model with best params")
from LRByHand import LinearRegressionByHand
r2_scores, mse_scores = np.array([]), np.array([])
model_by_hand = LinearRegressionByHand(learning_rate = 0.001, iterations = 100000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
losses = model_by_hand.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2, mse = r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

print('r2 : {}'.format( r2 ))
print('mse: {}'.format( mse ))
#print(losses)

#Testing deployed model
id = 1234   #selecting student
student_to_predict = df_training_dataset.iloc[ id ].to_numpy()[:-1].tolist() #drop the target column

for i in range(len(student_to_predict)):
    if type(student_to_predict[i]) == np.float64 or type(student_to_predict[i]) == np.int64 :
        student_to_predict[i] = student_to_predict[i].item()

print("student to predict: ")
print(student_to_predict)

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
#API_KEY = "-G3U50xfvNeOz0d8uCTPDIUN2kdr3fTOz-fczYAZqcHb"
API_KEY = "QVm0K54oNZSk1t82d1LKihn-sxQHL3oXSgfkYrJturZR"
token_response = requests.post('https://iam.ng.bluemix.net/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

fields = ['Unnamed: 0', 'EDAD', 'GENERO', 'RESIDENCIA', 'NV_ESTUDIO', 'ESTUDIO_PREV', 'TRACK_DH', 'AVG_DH', 'MINUTES_DH', 'EXPERIENCIA']

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"fields": fields, "values": [ student_to_predict ]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/14dcd737-84a8-4af9-a61f-68e079b29c4f/predictions?version=2020-11-23', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})

print("-------------------")

print("Scoring response from deployed model")
print(response_scoring.json()["predictions"][0]["values"][0][0])

print("Scoring response from manual model")
print(model_by_hand.predict(X[ id ]))

print("Expected response: ")
print(df_training_dataset.iloc[ id ]["DIAS_EMP"])

