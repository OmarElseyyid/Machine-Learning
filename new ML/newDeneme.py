import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x = x[:, 1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression 
le = LinearRegression()
le.fit(x_train, y_train)

y_pred = le.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

print(score)

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, le.predict(x_train), color="blue")
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)


plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color="blue")
plt.show()
print(lin_reg.predict(poly_reg.fit_transform([[10]])))