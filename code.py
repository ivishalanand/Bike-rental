# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 02:32:52 2019

@author: vishal
"""
import pandas as pd
import numpy as np

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

X=pd.concat([train,test],axis=0, sort=False)

X.describe()
X.info()
X.isnull().sum()
X=X.drop(["casual","registered"],axis=1)
#take out y
y_train=X.iloc[:train.shape[0],-1:]
X=X.drop(["count"],axis=1)

datetime=X.datetime.iloc[train.shape[0]:]
X=X.drop(['datetime'],axis=1)

X_numeric=X.select_dtypes(exclude="object")

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_numeric=pd.DataFrame(scaler.fit_transform(X_numeric),columns=X_numeric.columns,index=X_numeric.index)

X_train=X_numeric.iloc[:train.shape[0]]
X_test=X_numeric.iloc[train.shape[0]:]

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
    
y_pred=pd.DataFrame(classifier.predict(X_test),columns=["count"])

df=pd.concat([datetime,y_pred],axis=1)

df.to_csv("submissions.csv",index=False,header=["datetime","count"])
