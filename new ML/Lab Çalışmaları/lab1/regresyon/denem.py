# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dataset = pd.read_csv('SirketVerileri.csv', encoding = 'iso-8859-9')


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

print(x)

from sklearn.preprocessing import Imputer
inputer = Imputer(missing_values= np.nan, strategy= "mean" , axis=0)