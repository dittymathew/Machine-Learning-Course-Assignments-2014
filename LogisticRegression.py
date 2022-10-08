import numpy as np
from sklearn.preprocessing import normalize
from sklearn import  linear_model
from sklearn.metrics import classification_report
from sklearn import metrics

#3500 * 1897
training_data = np.loadtxt(open('data/contest-only-train.csv',"rb"),delimiter=",")

Y_train = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

X_norm = normalize(X_train, norm='l2', axis=1)

#############################    Data is processed ############################################

clf = linear_model.LogisticRegression() # Logistic Regression
clf.fit(X_norm, Y_train)
Y_pred = clf.predict(X_norm)
"""
sio.mmwrite("train_x", X_norm)
sio.mmwrite("train_b", np.matrix(Y_train).T)
os.popen("l1_logreg_train -s train_x.mtx train_b.mtx 0.01 model") #Boyd's code

sio.mmwrite("test_x", X_norm)
sio.mmwrite("test_b", np.matrix(Y_train).T)
os.popen("l1_logreg_classify -t test_b.mtx model test_x.mtx results")

os.popen("l1_logreg_classify model test_x.mtx result")

target_names = ['class1', 'class2']
print(classification_report(Y_train, Y_pred, target_names=target_names))

#predict using l1_regularized logistic regression
Y_pred = sio.mmread('result')

"""
target_names = ['class1', 'class2']
print(classification_report(Y_train, Y_pred, target_names=target_names))
