#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
dataset = pd.read_csv('dataset_name.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

