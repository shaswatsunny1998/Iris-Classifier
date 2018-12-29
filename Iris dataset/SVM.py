#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

#Importing dataset
dataset=pd.read_csv('iris.csv')


#Data Preprocessing
X=dataset.iloc[:,1:5].values
y=dataset.iloc[:,5].values
le=LabelEncoder()
y=le.fit_transform(y)


#Splitting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Training Classifiers
#Training Linear classifier
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
print("The accuracy of the linear kernel model is:",accuracy_score(y_test,svc.predict(X_test)))
print("The confusion matrix of the linear model is: ",confusion_matrix(y_test,svc.predict(X_test)))

#Training Ploynomial classifier
svc=SVC(kernel='poly',degree=5)
svc.fit(X,y)
print("The accuracy of the polynomial model is:",accuracy_score(y_test,svc.predict(X_test)))
print("The confusion matrix of the polynomial model is:",confusion_matrix(y_test,svc.predict(X_test)))

