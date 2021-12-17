# -*- coding: utf-8 -*-


# Support Vector Machine (Destek Vektör Makineleri)

#Kütüphanelerin eklenmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling

#Veri setinin import edilmesi
dataset = pd.read_csv('sigorta.csv', encoding = 'iso-8859-9')

#data_profile = pd.read_csv('sigorta.csv', encoding = 'iso-8859-9').profile_report()

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values
y = y.reshape(-1, 1) 



# Kategorik verilerin sayısallaştırılması
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1]) 

X[:, 4] = labelencoder.fit_transform(X[:, 4])

X[:, 5] = labelencoder.fit_transform(X[:, 5])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Veri setinin eğitim ve test olarak bölümlenmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Özellik ölçekleme (Feature Scaling)
# SVR algoritması ölçeklenmiş veri ister !
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
y_train_scale = scaler.fit_transform(y_train) 
y_test_scale = scaler.fit_transform(y_test) 


# SVR fit : SVR parametreleri derste anlatildi biz varsayalini kullanacagiz
# parameter tuning de her parametre icin ornegin GridsearchCV ile optimizasyon yapilabilir..
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_scale, y_train_scale)
y_pred_scale = regressor.predict(X_test_scale)


from sklearn.metrics import r2_score
r2_score(y_test_scale, y_pred_scale)

y_pred_scale = y_pred_scale.reshape(-1, 1)
y_predGercek = scaler.inverse_transform(y_pred_scale)


#verileri ölçeklemeden modeli eğitelim ve test edelim
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_predOlceksiz = regressor.predict(X_test)

#r2 deðerinin veriler ölçeksiz olduğundaki durumunu görelim.
#r2 deðerinin negatif çıkması modelin başarısının çok kötü olduğu anlamına gelmektedir.
from sklearn.metrics import r2_score
r2_score(y_test, y_predOlceksiz)
 