#Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

#Importing dataset
dataset=pd.read_csv('iris.csv')


#Dataset Preprocessing
X=dataset.iloc[:,1:5].values
y=dataset.iloc[:,5].values
label=LabelEncoder()
y=label.fit_transform(y)


#Training classifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
print("The accuracy of the model is: ",accuracy_score(y_test,knn.predict(X_test))*100,end='%')
print("The confusion matrix is: ",confusion_matrix(y_test,knn.predict(X_test)))
