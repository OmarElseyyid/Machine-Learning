# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

veriset = pd.read_csv("odev_tenis.csv")
print(veriset)

x = veriset.iloc[:,[0,1,3,4]] 
y = veriset.iloc[:,2].astype(np.float64)

print(x)
print(y)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,-1] = le.fit_transform(x[:,-1])
x[:,-2] = le.fit_transform(x[:,-2])


print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


from sklearn.metrics import r2_score
sonuc = r2_score(y_test, y_pred)

print(sonuc)

import statsmodels.api as sm
x = np.append(arr= np.ones((14,1)).astype(int), values=x, axis=1)
x_opt = x[:,[0,1,2,3,4]]
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()