import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv", sep=",")
print(df.head(8))
print(df.columns)
print(df.dtypes)

#we can see it is similar to Logistic Function.
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values) #your X and Y
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

"""Sigmoid Function"""
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

"""Model"""
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2) #call function

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000., 'b')
plt.plot(x_data, y_data, 'ro')
plt.show()

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)
#print(len(xdata),max(x_data))

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

###################################Model data using sklearn #################################
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )