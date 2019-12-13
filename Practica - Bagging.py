# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:32:42 2019

@author: irodr
"""


#Paquetes
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import pandas as pd
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

#Cargamos dataset y revisamos variables
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data.keys()
for i in data.keys():
    print(i)
    print(data[i][0:5])

#Como es un diccionario, transformamos en un dataframe
dt=pd.DataFrame(data['data'],columns=data['feature_names'])
dt['Diagnosis']=np.where(data.target==1, "Malignant","Bening")
dt.columns
dt.shape #30 variables predictoras

#Definimos target y dataset predictor
Y=data['target']
X=dt.drop(['Diagnosis'],axis=1)
Y.head()
X.columns

#Creamos modelo
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
y_train[y_train==1].count()/y_train.shape[0]*100
y_test[y_test==1].count()/y_test.shape[0]*100

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(presort= True)
model.get_params()
params={'criterion':['gini','entropy'],
        'max_depth': [30,40,50],# Maxima pofundidad del arbol
        'max_features': [20, 30], # numero de features a considerar en cada split
        'max_leaf_nodes': [10,20,30], # maximo de nodos del arbol
        'min_impurity_decrease' : [0.05], # un nuevo nodo se harà si al hacerse se decrece la impurity en un threshold por encima del valor
        'min_samples_split': [5,10] # The minimum number of samples required to split an internal node:
        }

scoring = ['accuracy', 'roc_auc']
n = 2
dispath=None
n_cv=5
grid_solver = GridSearchCV(estimator = model, # model to train
                   param_grid = params, # param_grid
                   scoring = scoring,
                   cv = n_cv,
                   refit = 'roc_auc',# cuando tienes mas de un score para tener un criterio de cual es el mejor modelo
                   verbose = 2)

# Hacemos cross validation con grid_solver
model_result = grid_solver.fit(X_train,y_train)
# best score es la media del best estimator
model_result.best_score_
# tenemos los mejores parametros
model_result.best_params_

# tenemos un atributo del mejor modelo
best_model=model_result.best_estimator_
# nos quedamos con el mejor modelo
final_model=best_model.fit(X_train,y_train)
final_model.predict_proba(X_train)
final_model.predict(X_train)
final_model.score(X_train,y_train)

final_model.predict_proba(X_test)
final_model.predict(X_test)
final_model.score(X_test,y_test)


#Una vez tenemos el model_result con los best_params_ 
#calculamos nuestro modelo final APLICANDO BAGGIN para disminuir la varianza
import statistics
import random
import numpy as np

def bagging(resamples):
    bag=[]
    for j in range(1,resamples):
        X_muestra,y_muestra=sklearn.utils.resample(X_train,y_train,replace=True)
        #X_muestra=muestra.drop(['Diagnosis'],axis=1)
        model = DecisionTreeClassifier(presort= True)
        model.set_params(**model_result.best_params_)
        model.fit(X_muestra,y_muestra)
        bag.append(model)
#        prob=final_model.predict_proba(X_muestra)
#        prob_list.append(prob)
#        prob_array=np.asarray(prob_list)
#    np.mean(prob_array,axis=0)
#    return np.mean(prob_array,axis=0)
    return bag

bag=bagging(100)
def pred_bagging(bag,X_test,y_test,threshold=0.5):
    print(bag[0].classes_)
    preds=[]
    for i in range(len(bag)):
        pred=bag[i].predict_proba(X_test)
        preds.append(pred)
    preds_array=np.asarray(preds)
    y_probs=np.mean(preds_array,axis=0)
    y_preds=np.where(y_probs<threshold,0,1)
    print(sklearn.metrics.accuracy_score(y_test,y_preds[:,1]))
    return(y_preds)


probs_mean=bagging(100)
probs_mean
final_model.classes_ #Para saber las columnas de las probabilidades

y_hat=[]
for i in range(len(probs_mean)):
    if probs_mean[i][1]>=0.5:
        y_hat.append('Malignant')
    elif probs_mean[i][1]<0.5:
        y_hat.append('Bening')
y_hat


sns.distplot()
