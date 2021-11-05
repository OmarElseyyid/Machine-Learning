#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
dataset = pd.read_csv('dataset_name.csv')
#split the dataset to x and y
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encode the categorical values with OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [ the column number we want to change ])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#encode the y categorical values with LableEncoder
from sklearn.preprocessing import LableEncoder
le = LableEncoder()
y = le.fit_transform(y)

#split dataset to x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

    #predict test values 
    y_pred = regressor.predict(x_test)
    
    #calculation of error difference between test values and prediction values. The part where the success of the model is measured.
    from sklearn.metrics import r2_score
    score= r2_score(y_test, y_pred)

    #show test prediction values
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.show()
    
