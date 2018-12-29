#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

#Training Multinomial Classifier
multi=MultinomialNB()
multi.fit(X_train,y_train)
print("The accuracy of the multinomial bayes theorem is:",accuracy_score(y_test,multi.predict(X_test)))
print("The consfusion matrix of the model is:",confusion_matrix(y_test,multi.predict(X_test)))