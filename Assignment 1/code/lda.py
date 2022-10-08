# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 18:17:33 2014

@author: ditty
"""

#import matplotlib.pyplot as plt
from sklearn import metrics
#from mpl_toolkits.mplot3d import Axes3D

#from matplotlib.mlab import PCA as mlabPCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.lda import LDA


def evaluate(Y_test,Y_predict):
  print Y_test.shape,Y_predict.shape
  acc =metrics.accuracy_score(Y_test,Y_predict)
  precision =metrics.precision_score(Y_test,Y_predict)
  recall =metrics.recall_score(Y_test,Y_predict)
  fmeasure =metrics.f1_score(Y_test,Y_predict)
  print metrics.confusion_matrix(Y_test,Y_predict)
  print 'Accuracy =',acc
  print 'Precision =',precision
  print 'Recall =',recall
  print 'F-measure =',fmeasure

def getData(f1,f2,f3,f4):
  X_train =  np.loadtxt(open(f1,"rb"), delimiter = ',')
  X_test =  np.loadtxt(open(f2,"rb"), delimiter = ',')
  Y_train =  np.loadtxt(open(f3,"rb"), delimiter = ',')
  Y_test =  np.loadtxt(open(f4,"rb"), delimiter = ',')
  return X_train,X_test,Y_train,Y_test

np.random.seed(5)

x_train,x_test,y_train,y_test = getData("DS3/train.csv","DS3/test.csv","DS3/train_labels.csv","DS3/test_labels.csv")
X =np.concatenate((x_train,x_test))
y= np.hstack((np.matrix(y_train),np.matrix(y_test))).T
centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
lda = LDA(n_components=1)
lda.fit(x_train,y_train)
transformed = lda.transform(X)
print x_train.shape, transformed.shape
#t_x_train =np.matrix(transformed[0:x_train.shape[0],0]).T
#t_x_test =np.matrix(transformed[x_train.shape[0]:,0]).T
t_x_train =transformed.T[0:x_train.shape[0]]
t_x_test =transformed.T[x_train.shape[0]:]
print t_x_train.shape, t_x_test.shape
t_x =np.row_stack((t_x_train,t_x_test))
#y_train =np.matrix(y_train)
y_test =np.matrix(y_test)
print t_x_train.shape,t_x_test.shape,y_train.shape,y_test.shape
y_predict=[]
for i in range(0,y_test.shape[1]):
  predict = lda.predict(x_test[i])
  y_predict.append(predict)
print y_predict
#y_predict = regClassify(t_x_train,t_x_test,y_train,y_test)
print y_test ,y_predict
evaluate(y_test.T,np.matrix(y_predict))
#print y[:,0]
classA=[]
classB=[]
t_classA=[]
t_classB=[]
print t_x.shape
for i in range(X.shape[0]):
   
    if y[i,:] ==1:
        classA.append(X[i])
        t_classA.append(t_x[:,i])
    else:
        classB.append(X[i])
        t_classB.append(t_x[:,i])

classA= np.array(np.matrix(classA).T)
classB= np.array(np.matrix(classB).T)
t_classA=np.array(t_classA)
t_classB=np.array(t_classB)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

#plt.scatter(X[:,0],X[:,1],X[:,2],marker ='+')
ax.plot(classA[0],classA[1],classA[2],c='red',marker ='o')
ax.plot(classB[0],classB[1],classB[2],c='green',marker ='o')
plt.show()

fig =plt.figure()
#plt.scatter(X[:,0],X[:,1],X[:,2],marker ='+')
plt.scatter(t_classA[:,0],np.ones(t_classA.shape[0]),c='red',marker ='o')
plt.scatter(t_classB[:,0],np.ones(t_classB.shape[0]),c='green',marker ='o')
plt.show()
