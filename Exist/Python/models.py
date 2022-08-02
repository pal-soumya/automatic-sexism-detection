#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:53:48 2021

@author: soumya
"""
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def random_forest(X_train,y_train,X_test):
    model = RandomForestClassifier(n_estimators=1000, max_depth=150,n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def svm(X_train,y_train,X_test,kernel):
    print("i am priicting")
    svclassifier = SVC(kernel=kernel)  
    svclassifier.fit(X_train, y_train)  
    y_pred = svclassifier.predict(X_test)  
    print("i pridicted")
    return y_pred

def KNN(X_train,y_train,X_test):
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    return y_pred