import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

colnames = [label.strip() for label in open("columns.names").readline().rstrip().split(',')]
bdata = read_csv("housing.data", sep="\s+", header=None, names=colnames)

pd.set_option('display.precision', 2)
#print(bdata.corr(method='pearson'))

# selecting the following columns from the panda dataframe :INDUS, NOX, RM, TAX, PTRATIO, LSTAT, MEDV
clean_data = bdata[['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']]
#print(clean_data)



features = clean_data.drop('MEDV', axis = 1)
prices = clean_data['MEDV']
#print(features)
#print(prices)

num_pipeline  = Pipeline([('std_scaler', StandardScaler())])
lin_regressor = num_pipeline.fit(features,prices)

lin_reg = LinearRegression()
lin_reg.fit(features,prices)

data
#LinearRegression().fit(features,prices)
#features = clean_data.drop('MEDV', axis = 1)
#prices = clean_data['MEDV']