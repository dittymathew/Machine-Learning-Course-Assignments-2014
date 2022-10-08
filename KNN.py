import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
from numpy.linalg import inv
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

training_data = np.loadtxt(open('contest-only-train.csv',"rb"),delimiter=",")

Y_train = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

clf = KNeighborsClassifier(n_neighbors=500)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_train)


print metrics.accuracy_score(Y_train, Y_pred)
print metrics.precision_score(Y_train, Y_pred)
print metrics.recall_score(Y_train, Y_pred)
print metrics.f1_score(Y_train, Y_pred)

target_names = ['class1', 'class2']
print(classification_report(Y_train, Y_pred, target_names=target_names))
