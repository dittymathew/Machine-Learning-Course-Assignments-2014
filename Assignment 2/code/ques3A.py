import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import normalize
import scipy.io as sio
import os

category ={'coast':1,'forest':2,'insidecity':3,'mountain':4}
train =np.loadtxt(open("DS2-train.csv","rb"), delimiter=",")
test = np.loadtxt(open("DS2-test.csv","rb"), delimiter=",")
indices = [i for i in range(0,1006) if train[i,96] in [2,4]]
train = train[indices]
x_train=train[:,0:96]
y_train=train[:,96]
for i in range(0,len(y_train)):
  if y_train[i] ==2:
    y_train[i]=-1
  else:
    y_train[i]=1
x_train = normalize(x_train, norm='l2', axis=1)
indices = [i for i in range(0,80) if test[i,96] in [2,4]]
test = test[indices]
x_test=test[:,0:96]
y_test=test[:,96]
for i in range(0,len(y_test)):
  if y_test[i] ==2:
    y_test[i]=-1
  else:
    y_test[i]=1

x_test = normalize(x_test, norm='l2', axis=1)
h = .02  

logreg = linear_model.LogisticRegression(C=1)

logreg.fit(x_train,y_train)

y_predict =logreg.predict(x_test)
print y_predict

print metrics.confusion_matrix(y_test,y_predict)
print metrics.classification_report(y_test,y_predict)
sio.mmwrite("train_x", x_train)
sio.mmwrite("train_b", np.matrix(y_train).T)
sio.mmwrite("test_x", x_test)
sio.mmwrite("test_b", np.matrix(y_test).T)
os.popen("l1_logreg_train -s train_x.mtx train_b.mtx 0.001 model")
os.popen("l1_logreg_classify model test_x.mtx result")
y_predict = sio.mmread('result')
y_predict =[y[0] for y in y_predict ]
print y_predict
print y_test
print metrics.confusion_matrix(y_test,y_predict)
print metrics.classification_report(y_test,y_predict)
