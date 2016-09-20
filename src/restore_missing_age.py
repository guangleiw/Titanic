#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("/Users/wangguanglei/dev/py_workspace/Titanic/data/train.csv")

def restore_missing_age(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    #print(type(known_age))

    y = known_age[:,0]
    #print(y)
    X = known_age[:,1:]
    #print(X)

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)


    predictedAge = rfr.predict(unknown_age[:,1::])

    df.loc[(df.Age.isnull()),'Age'] =predictedAge

    return df,rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


data_train, rfr = restore_missing_age(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df)