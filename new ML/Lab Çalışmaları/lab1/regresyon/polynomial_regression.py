# -*- coding: utf-8 -*-

# Polynomial Regression

# Kütüphanelerin eklenmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri setinin okunarak bağımlı ve bağımsız değişkenlere ayrılması
dataset = pd.read_csv('PozisyonMaas.csv', encoding = 'iso-8859-9')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Veri setinin boyutu küçük olduğu için eğitim ve test olarak ayırmıyoruz.
#Tüm veri setini kullanıyoruz. 


# modelin basit regresyon ile eğitilmesi
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# modelin polinom regresyon ile eğitilmesi
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# basit regresyon sonuçlarının gösterimi
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Pozisyon')
plt.ylabel('Maaş')
plt.show()

# polinom regresyon sonuçlarının gösterimi
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression)')
plt.xlabel('Pozisyon')
plt.ylabel('Maaş')
plt.show()

# Verilerin basit lineer ile test edilmesi
L= lin_reg.predict(X)

# Verilerin polinıom lineer ile test edilmesi
P=lin_reg_2.predict(X_poly)

#karşılaştırma
LineerKarsilastirma = pd.DataFrame( X, columns = ['Seviye'])
LineerKarsilastirma = LineerKarsilastirma.assign(GercekMaaş = y) 
LineerKarsilastirma = LineerKarsilastirma.assign(LineerTahmin = L) 
LineerKarsilastirma = LineerKarsilastirma.assign(PolinomTahmin = P)