import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# When data is non linear we use Polynomial Regression

df = pd.read_csv("FuelConsumption.csv", sep=",")
print(df.shape) #(row, columns)

"""split data in test and train"""
mask = np.random.rand(len(df)) <0.8
train = df[mask]
test = df[~mask]

"""Plot Enginesize(X) vs co2Emission(Y)"""
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2Emission")
plt.show() # make sure tkinter or pyQt is installed if you not on notebook.


train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

"""Polynomial Feature (preprocessing)"""
from sklearn.preprocessing import PolynomialFeatures
# if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2
#poly = PolynomialFeatures(degree=2) #In short we are using multi regression, one features is used, but with 3 thetas means we created additional features.
poly = PolynomialFeatures(degree=3)
train_x_transformed = poly.fit_transform(train_x)

"""Model Data and Fit line"""
from sklearn import linear_model
poly_non = linear_model.LinearRegression()
poly_non.fit(train_x_transformed, train_y)
print(f'Intercept : {poly_non.intercept_}')
print(f'Coefficients : {poly_non.coef_}')

"""Plot fit line"""
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)#(start,stop,step)
#yy = poly_non.intercept_[0]+ poly_non.coef_[0][1]*XX+ poly_non.coef_[0][2]*np.power(XX, 2)
yy = poly_non.intercept_[0]+ poly_non.coef_[0][1]*XX + poly_non.coef_[0][2]*np.power(XX, 2) + poly_non.coef_[0][3]*np.power(XX, 3)
#print(len(yy),len(XX))
plt.plot(XX, yy, '-r' ) #plot fit line graph
plt.xlabel("Engine Size")
plt.ylabel("CO2Emission")
plt.show()

"""Evaluation"""
test_x = np.asanyarray(test[['ENGINESIZE']]) #2D array [[]]
test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_x_poly = poly.transform(test_x) #poly transform degree 2, use simple tranform not for fit
test_pred_y = poly_non.predict(test_x_poly)

from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_pred_y - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_pred_y - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_pred_y) )
#change value of degree to see its effect on results