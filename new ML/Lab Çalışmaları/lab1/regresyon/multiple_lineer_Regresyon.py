# -*- coding: utf-8 -*-


# Multiple Linear Regression (Çoklu Doğrusal(Lineer) Regresyon)

# Kütüphanelerin eklenmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri setinin okunarak bağımlı ve bağımsız değişkenlere ayrılması
dataset = pd.read_csv('SirketVerileri.csv', encoding = 'iso-8859-9')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
y = y.reshape(-1, 1) 

# Kategorik verilerin sayısallaştırılması

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Veri setinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Özellik ölçekleme
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.fit_transform(y_test) 

# Eğitim verileri ile modelin eğitilmesi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Test verileri kullanılarak tahmin yapılması
y_pred = regressor.predict(X_test)

#Test değerleri ile tahmin değerleri arasındaki hata farkının hesaplanması
#Modelin başarısının ölçüldüğü kısım.
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


