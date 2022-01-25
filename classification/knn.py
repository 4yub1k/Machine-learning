import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read this article, Must
#https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp


#custcat is the 4 categories in which customer are divided
#depends on 'region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside'
#we will predict a category for new customer
#x=features, y=labels
#category to numeric conversion using pandas
#https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
"""Read csv"""
df = pd.read_csv("teleCust_knn.csv", sep=",")
#print(df.head())
print(f'Number of row : {df.shape[0]}')
print(f'Custcat/categories : \n{df["custcat"].unique()}')
print(f'Number of row in each Custcat/categories : \n{df["custcat"].value_counts()}')
print(f'Number of columns : {len(df.columns)}')
#print(df.columns) #print columns

"""Values"""
#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
#use if using built in test/train 
x=df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
y=df["custcat"].values #without any index, only an array of values index wise
#print(y[0:5])
#print(df.index.values)
colors = {1:'red', 2:'green', 3:'blue', 4:'yellow'}
#z=df['custcat'].squeeze() #series conversion as map takes series
#print(z.map(colors))
#df['age'].iloc[0:100] #range of rows

#uncomment to see scatter graph
plt.scatter(df['age'].iloc[0:60], df['income'].iloc[0:60], c=df['custcat'].iloc[0:60].map(colors))
plt.show()

"""Test/Train"""
#x=train, y=test
mask = np.random.rand(len(df)) < 0.8 #len(df)== numbers of rows. we are generating true,false for them.
#print(mask) #msk true if numbers generated by np is < 0.8 else >=0.8 False
train = df[mask] # 80% , it pick the row for which mask is true.
test = df[~mask] #~ for remaining 20%, for mask false

print(train.head())
print(test.head())
"""Test/train using sklearn"""
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

train_x=np.asanyarray(train[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']]) #2D array [[]]
train_y=np.asanyarray(train[['custcat']])

test_x=np.asanyarray(test[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']]) #2D array [[]]
test_y=np.asanyarray(test[['custcat']])

# """Pre processing"""
#Data Standardization gives the data zero mean and unit variance, 
#it is good practice, especially for algorithms such as KNN which is based on the distance of data points:
# from sklearn import preprocessing
# train_x = preprocessing.StandardScaler().fit(train_x).transform(train_x.astype(float))

"""Model data sklearn"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#select best value for k
k_values={}
for k in range(1,20):
    #k = 6 #number of neighbours
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = k).fit(train_x,train_y)
    test_pred_y=neigh.predict(test_x)
    k_values[k]=metrics.accuracy_score(test_y, test_pred_y) #Test set accuracy
k=max(k_values,key=k_values.get) #get key of max value
print("\n",k_values,"\n")
print("NOTE : very high value of K will lead to Over Generalized")
print("Best value for K : ",k,"\nTest set accuracy : ",k_values[k])
#print("Best value of K : ",max(k_values.values()))


"""Evaluation"""

#from sklearn import metrics
#Accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function
#print("Train set Accuracy: ", metrics.accuracy_score(train_y, neigh.predict(train_x)))
#print("Test set Accuracy: ", metrics.accuracy_score(test_y, test_pred_y))

"""Plot with best K"""
plt.scatter(k_values.keys(),k_values.values(), c='green')
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()