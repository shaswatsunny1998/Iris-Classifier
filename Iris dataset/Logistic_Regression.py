#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#Importing dataset
dataset=pd.read_csv("iris.csv")




#Splitting dataset
X=dataset.iloc[:,1:5].values
y=dataset.iloc[:,5].values
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)


#Standardization

#Training models
logic=LogisticRegression(C=3.0,solver="liblinear",random_state=10,max_iter=10000)
logic.fit(X_train,y_train)
print("The accuracy percent of classifier on trained data is: ",accuracy_score(y_train,logic.predict(X_train)))
print("The accuracy percent of classifier on test data is: ",accuracy_score(y_test,logic.predict(X_test)))

#K-Cross validation
cross=cross_val_score(estimator = logic, X = X_train , y = y_train , cv=10 ,n_jobs=-1)


#Grid Search
parameters={"C":[3.0,3.5,3.9,5.0],"solver":["newton-cg","lbfgs","liblinear","sag","saga"],}
grid=GridSearchCV(estimator=logic,param_grid=parameters,scoring="accuracy",n_jobs=-1,cv=10)
grid=grid.fit(X_train,y_train)


#EDA
'''
plt.figure(1)
plt.hist(dataset.iloc[:,1])
plt.title("Sepal length")
plt.show()
plt.figure(2)
plt.hist(dataset.iloc[:,2])
plt.title("Sepal width")
plt.show()
plt.figure(3)
plt.hist(dataset.iloc[:,3])
plt.title("Petal length")
plt.show()
plt.figure(4)
plt.hist(dataset.iloc[:,4])
plt.title("Petal width")
plt.show()
sns.heatmap(dataset.iloc[:,1:5])
largest_sepal_length=max(dataset.iloc[:,1])
smallest_sepal_length=min((dataset.iloc[:,1]))
largest_sepal_width=max(dataset.iloc[:,2])
smallest_sepal_width=min((dataset.iloc[:,2]))
largest_petal_length=max(dataset.iloc[:,3])
smallest_petal_length=min((dataset.iloc[:,3]))
largest_petal_width=max(dataset.iloc[:,4])
smallest_petal_width=min((dataset.iloc[:,4]))
'''