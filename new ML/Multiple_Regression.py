# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("veriler.csv")

x = dataset.iloc[:,:2].values
y = dataset.iloc[:,3].values

ct = ColumnTransformer([('encoder', preprocessing.OneHotEncoder() , [0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y, test_size = 0.2, random_state =0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score

rKare  = r2_score(y_test,y_pred)


print(rKare)