# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:19:19 2022

@author: SANJUSHA
"""

# DECISION TREE CLASSIFIER

import pandas as pd
import numpy as np

df=pd.read_csv("Fraud_check.csv")
df
df.isnull().sum()
# There are no null values
df.info()

# Boxplots
df.boxplot("City.Population",vert=False)
df.boxplot("Work.Experience",vert=False)
# There are no outliers

df["Taxable.Income"]=pd.cut(df["Taxable.Income"], bins=[0,30000,99620], labels=["Risky","Good"])
df

# Splitting the variables
Y=df["Taxable.Income"]

X1=df.iloc[:,:2]
X2=df.iloc[:,3:]
X=pd.concat([X1,X2],axis=1)
X.dtypes

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["Undergrad"]=LE.fit_transform(X["Undergrad"])
X["Undergrad"]=pd.DataFrame(X["Undergrad"])

X["Marital.Status"]=LE.fit_transform(X["Marital.Status"])
X["Marital.Status"]=pd.DataFrame(X["Marital.Status"])

X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["City.Population"]=MM.fit_transform(X[["City.Population"]])
X["Work.Experience"]=MM.fit_transform(X[["Work.Experience"]])
X

Y=LE.fit_transform(df["Taxable.Income"])
Y=pd.DataFrame(Y)


# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# Model fitting
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(max_depth=3)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)
Y_predtest=DT.predict(X_test)

# Metrics
from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain) # 0.804761
ac2=accuracy_score(Y_test,Y_predtest) # 0.783333
# If max_depth is 3 and test_size is 0.3 then ac1=80% and ac2=78%


DT.tree_.max_depth # number of levels = 3
DT.tree_.node_count # counting the number of nodes = 13


# Tree visualization
# pip install graphviz
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data, format="png")  
graph


# To know which is the best max depth value and max leaf node value we are doing gridesearchcv 
from sklearn.model_selection import GridSearchCV

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=DecisionTreeClassifier(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_
# If max_depth is 1 and max_leaf_nodes is 2 then best_score is 0.78571

# Entropy method
Training_accuracy = []
Test_accuracy = []

for i in range(1,12):
    classifier = DecisionTreeClassifier(max_depth=i,criterion="entropy") 
    classifier.fit(X_train,Y_train)
    Y_pred_train = classifier.predict(X_train)
    Y_pred_test = classifier.predict(X_test)
    Training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    Test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
    
pd.DataFrame(Training_accuracy)
pd.DataFrame(Test_accuracy)
pd.concat([pd.DataFrame(range(1,12)) ,pd.DataFrame(Training_accuracy),pd.DataFrame(Test_accuracy)],axis=1)    

# Best Training_accuracy=89% and Best Test_accuracy=82%
