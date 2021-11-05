#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
dataset = pd.read_csv('dataset_name.csv')
#split the dataset to x and y
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #strategy types:  {"mean" , "median", "most_frequent", "constant"}
imputer.fit(x[:, the columns number we want to change ])
X[:, the column number we want to change ] = imputer.transform(X[:, the column number we want to change ])

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

#features scaling
   #using MinMax
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   x_train_scale = scaler.fit_transform(x_train)
   x_test_scale = scaler.fit_transform(x_test)
   y_train_scale = scaler.fit_transform(y_train) 
   y_test_scale = scaler.fit_transform(y_test) 
    
   #using StandardScaler
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()    
   x_train_scale = sc.fit_transform(x_train)
   x_test_scale = sc.fit_transform(x_test)
   y_train_scale = sc.fit_transform(y_train) 
   y_test_scale = sc.fit_transform(y_test) 

   #inverse scale after prediction
   scaler.inverse_transform(prediction command)
   sc.inverse_transform(prediction command)


# 1) -------------------------------------------------
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
    
# 2) -------------------------------------------------
#multiple linear regression
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

# 3) -------------------------------------------------
#polynomial regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    #teach the model with polynomial regression features
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 3) #choose the degree
    x_poly = poly_reg.fit_transform(x)
    poly_reg.fit(x_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y)

        #show model prediction values
            #linear model
            plt.scatter(x, y, color = 'red')
            plt.plot(x, lin_reg.predict(X), color = 'blue')
            plt.show()
            #poly model
            plt.scatter(x, y, color = 'red')
            plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
            plt.show()

        #test the values with simple linear 
        L= lin_reg.predict(X)

        #test the values with poly linear
        P=lin_reg_2.predict(X_poly)

        #compare the 2 models
        LineerKarsilastirma = pd.DataFrame( x, columns = ['Seviye'])
        LineerKarsilastirma = LineerKarsilastirma.assign(GercekMaaş = y) 
        LineerKarsilastirma = LineerKarsilastirma.assign(LineerTahmin = L) 
        LineerKarsilastirma = LineerKarsilastirma.assign(PolinomTahmin = P)

# 4) -------------------------------------------------
#support vector machine model (SVM)
#SVR support vector regressor

    #with scaling -----
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf') #we have a deffirent types of kernel {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
    regressor.fit(x_train_scale, y_train_scale)
    y_pred_scale = regressor.predict(x_test_scale)

        #calculation of error difference between test values and prediction values. The part where the success of the model is measured.
        from sklearn.metrics import r2_score
        r2_score(y_test_scale, y_pred_scale)

        y_pred_scale = y_pred_scale.reshape(-1, 1)
        y_pred_normal = scaler.inverse_transform(y_pred_scale)

    #without scaling ----
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(x_train, y_train)
    y_predOlceksiz = regressor.predict(x_test)

        #calculation of error difference between test values and prediction values. The part where the success of the model is measured.
        from sklearn.metrics import r2_score
        r2_score(y_test, y_predOlceksiz)
        
# 5) -------------------------------------------------
#decision tree regressor
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state= 0)
    regressor.fit(x,y) 
            #predict the values
            regressor.predict([[ value ]])

        
# 6) -------------------------------------------------
#random forest regressor
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state = 0)











    
