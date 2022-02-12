import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

lin_regressor = Pipeline([('std_scaler', StandardScaler()),("lin_reg", LinearRegression())])

training_set = []
validation_set = []
set_size = []

for i in range (1,80):
    X_train, X_val, y_train, y_val = train_test_split(features, prices, test_size=(i/100))
    lin_regressor.fit(X_train, y_train)
    y_train_predict = lin_regressor.predict(X_train)
    y_val_predict = lin_regressor.predict(X_val)
    training_set.append(np.sqrt(mean_squared_error(y_train, y_train_predict)))
    validation_set.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))
    set_size.append(i)

plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.plot(set_size,validation_set,'r',set_size,training_set,'b')
plt.show()
#print(training_set)
#print(validation_set)

# Checking residuals
y_train_predict = lin_regressor.predict(X_train)
plt.scatter(y_train_predict, y_train-y_train_predict)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()