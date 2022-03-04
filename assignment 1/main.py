import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures # exercise 4
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


colnames = [label.strip() for label in open("columns.names").readline().rstrip().split(',')]
bdata = read_csv("../.ipynb_checkpoints/housing.data", sep="\s+", header=None, names=colnames)

pd.set_option('display.precision', 2)
#print(bdata.corr(method='pearson'))

#excerccise 1
# selecting the following columns from the panda dataframe :INDUS, NOX, RM, TAX, PTRATIO, LSTAT, MEDV
clean_data = bdata[['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']]
#print(clean_data)



#features = clean_data.drop('MEDV', axis = 1)

#prices = clean_data['MEDV']
#print(features)
#print(prices)

#excerccise 2
#lin_regressor = Pipeline([('std_scaler', StandardScaler()),("lin_reg", LinearRegression())])

#excerccise 3
#training_set = []
#validation_set = []
#set_size = []

#for i in np.arange (0.01,80,0,00.1):
    #X_train, X_val, y_train, y_val = train_test_split(features, prices, test_size=(i/100))
    #lin_regressor.fit(X_train, y_train)
    #y_train_predict = lin_regressor.predict(X_train)
    #y_val_predict = lin_regressor.predict(X_val)
    #training_set.append(np.sqrt(mean_squared_error(y_train, y_train_predict)))
    #validation_set.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))
    #set_size.append(i)

#plt.xlabel('Training set size')
#plt.ylabel('RMSE')
#plt.plot(set_size,validation_set,'r',label='val')
#plt.plot(set_size,training_set,'b',label='train')
#plt.legend(loc="upper right")
#plt.show()
#print(training_set)
#print(validation_set)

# Checking residuals
#y_train_predict = lin_regressor.predict(X_train)
#plt.scatter(y_train_predict, y_train-y_train_predict)
#plt.xlabel("Predicted")
#plt.ylabel("Residuals")
#plt.show()



#exercise 4
new_data = bdata.assign(CRIM=lambda x: np.log(x.CRIM),
                      NOX = lambda x: np.log(x.NOX),
                      DIS = lambda x: np.log(x.DIS),
                      LSTAT = lambda x: np.log(x.LSTAT))


prices = new_data['MEDV']
features = new_data[['LSTAT', 'PTRATIO', 'RM', 'INDUS', 'TAX']]


poly_ridge_pipeline = Pipeline([
 ("poly_features", PolynomialFeatures(degree=1, include_bias=False)),
 ("std_scaler", StandardScaler()),
 ("Lasso", Lasso())
 ])

training_set = []
set_size = []
validation_set = []

X_train, X_val, y_train, y_val = train_test_split(features, prices, test_size=(0.2))

for i in range(2,len(X_train)):
    poly_ridge_pipeline.fit(X_train[:i], y_train[:i])
    y_train_predict = poly_ridge_pipeline.predict(X_train[:i])
    y_val_predict = poly_ridge_pipeline.predict(X_val)
    training_set.append(np.sqrt(mean_squared_error(y_train[:i], y_train_predict)))
    validation_set.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))
    set_size.append(i)

plt.axis([0, 80,0, 10])
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.plot(set_size,validation_set,'r',label='val')
plt.plot(set_size,training_set,'b',label='train')
plt.legend(loc="upper right")
plt.show()