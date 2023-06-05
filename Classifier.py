# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:47:43 2023

@author: enesd
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



data = pd.read_csv("C:/Users/enesd/Downloads/diabetes.csv")
dt=data.copy()

y=dt["Outcome"]
X=dt.drop(columns="Outcome",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


def modelz(model):
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    skor=accuracy_score(y_test, pred)
    return round(skor*100,2)


models=[]
models.append(("Log Regression",LogisticRegression(random_state=0)))
models.append(("KNN",KNeighborsClassifier()))
models.append(("SVC",SVC(random_state=0)))
models.append(("Bayes",GaussianNB()))
models.append(("Decision Tree",DecisionTreeClassifier(random_state=0)))

modelname=[]
success=[]

for i in models:
    modelname.append(i[0])
    success.append(modelz(i[1]))


a=list(zip(modelname,success))
result=pd.DataFrame(a,columns=["Model","Score"])
print(result)
