import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Here we have pick the best drug (A,B) for a patient based on their age, sex etc

"""Load CSV"""
df = pd.read_csv("drug.csv", sep=',')
print(df[0:5])
print("Total row/entries : ",df.shape[0],df.shape)

"""Convert to array of vales (no column row)"""
x= df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y= df['Drug'] #.values will put it in array

"""Lets convert the categorical to numerical values pandas.get_dummies()"""
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M']) #name inside column, as 0,1
x[:,1] = le_sex.transform(x[:,1]) # age,sex,BP as sex is at 1 postion ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])#0,1,2
x[:,2] = le_BP.transform(x[:,2]) #at second position


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])#0,1
x[:,3] = le_Chol.transform(x[:,3]) #at third positions

print(x[0:5])
print(y[0:5])

"""Train/test split using built in"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

"""Model"""
from sklearn.tree import DecisionTreeClassifier
#max depth number of columns 0,1,2,3,4 | ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
#specify criterion="entropy"
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#drugTree # it shows the default parameters
drugTree.fit(x_train,y_train)

"""predict"""
#testing
y_pred = drugTree.predict(x_test)
print(y_pred[0:5]) #predicted y for test set
print(y_test[0:5]) #origional y test set, see the difference of prediction

"""Evaluation"""
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))
#as other tests take numerical values will use accuracy

"""To draw a tree save this snippet for future"""
from io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')