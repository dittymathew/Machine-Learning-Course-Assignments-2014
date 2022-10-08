import numpy as np
from sklearn.preprocessing import normalize
from sklearn import  linear_model
from sklearn.metrics import classification_report
import os
from sklearn import metrics

#3500 * 1897
training_data = np.loadtxt(open('data/contest-only-train.csv',"rb"),delimiter=",")

Y_train = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

X_norm = normalize(X_train, norm='l2', axis=1)

#############################    Data is processed ############################################

clf = linear_model.LinearRegression() # Linear Regression
clf.fit(X_norm, Y_train)
pred = clf.predict(X_norm)
Y_pred=[]
for val in pred:
  if val>=1.5:
    Y_pred.append(2)
  else:
    Y_pred.append(1)

target_names = ['class1', 'class2']
print(classification_report(Y_train, Y_pred, target_names=target_names))

