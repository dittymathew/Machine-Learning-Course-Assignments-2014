# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 18:17:33 2014

@author: ditty
"""

#import matplotlib.pyplot as plt
from sklearn import metrics
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

#from matplotlib.mlab import PCA as mlabPCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

def regClassify(X_train,X_test,Y_train,Y_test):

  X =np.c_[np.ones(X_train.shape[0]),np.matrix(X_train)]
  Y =np.matrix(Y_train)
  beta = np.linalg.inv(X.T*X).dot(X.T*Y.T)

  pclassA=[]
  pclassB=[]
  Y_predict =[]
  
  for i in range(0,Y_test.shape[1]):
   
    predict = np.c_[np.matrix(np.ones(1)),np.matrix(X_test[i])]*beta
    if predict <= 1.5:
      pclassA.append(list(X_test[i]))
      Y_predict.append(0)
    else:
      pclassB.append(list(X_test[i]))
      Y_predict.append(1)
  print X.shape

  return Y_predict

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
x_train,x_test,y_train,y_test = getData("DS2/train.csv","DS2/test.csv","DS2/train_labels.csv","DS2/test_labels.csv")
X =np.concatenate((x_train,x_test))
Y= np.hstack((np.matrix(y_train),np.matrix(y_test)))
meanX1 =np.mean(X[:,0])
meanX2 =np.mean(X[:,1])
meanX3 =np.mean(X[:,2])
meanX= np.array([[meanX1],[meanX2],[meanX3]])
s=np.zeros((3,3))

for i in range(X.shape[1]):
   
    s += (X[i,:]-meanX).dot(X[i,:]-meanX).T
print s
covariance =np.cov([X[0,:],X[1,:],X[2,:]])
#print covariance
eig_val, eig_vec = np.linalg.eig(covariance)
print eig_val
print eig_vec
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort()
eig_pairs.reverse()
matrix_w = eig_pairs[0][1].reshape(3,1)
print matrix_w
transformed = matrix_w.T.dot(X.T)
"""
t_x_train =transformed.T[0:x_train.shape[0]]
t_x_test =transformed.T[x_train.shape[0]:]
y_predict = regClassify(t_x_train,t_x_test,y_train,y_test)
evaluate(y_test,y_predict)
np.random.seed(5)

x_train,x_test,y_train,y_test = getData("DS3/train.csv","DS3/test.csv","DS3/train_labels.csv","DS3/test_labels.csv")
X =np.concatenate((x_train,x_test))
y= np.hstack((np.matrix(y_train),np.matrix(y_test)))
centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=1)
pca.fit(X)
transformed = pca.transform(X)
"""
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
y_predict = regClassify(t_x_train,t_x_test,y_train,y_test)
evaluate(y_test.T,np.matrix(y_predict).T)
#print y[:,0]
classA=[]
classB=[]
t_classA=[]
t_classB=[]
p_classA=[]
p_classB=[]
print t_x_test.shape
for i in range(len(y_predict)):
  if y_predict ==1:
    p_classA.append(t_x_test[i,:])
  else:
    p_classB.append(t_x_test[i,:])
for i in range(X.shape[0]):
   
    if Y[:,i] ==1:
        classA.append(X[i])
        t_classA.append(t_x[i])
    else:
        classB.append(X[i])
        t_classB.append(t_x[i])

classA= np.array(np.matrix(classA).T)
classB= np.array(np.matrix(classB).T)
t_classA=np.array(t_classA)
t_classB=np.array(t_classB)
p_classA=np.array(p_classA)
p_classB=np.array(p_classB)
print p_classA.shape
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

#plt.scatter(X[:,0],X[:,1],X[:,2],marker ='+')
ax.scatter(classA[0],classA[1],classA[2],c='red',marker ='o')
ax.scatter(classB[0],classB[1],classB[2],c='green',marker ='o')
plt.show()

#fig =plt.figure()
#plt.scatter(t_classA[:,0],np.ones(t_classA.shape[0]),c='red',marker ='o')
#plt.scatter(t_classB[:,0],np.ones(t_classB.shape[0]),c='green',marker ='o')
#plt.show()
fig =plt.figure()
plt.scatter(p_classA,np.ones(p_classA.shape[0]),c='red',marker ='o')
plt.scatter(p_classB,np.ones(p_classB.shape[0]),c='green',marker ='o')
plt.show()
"""
print transformed.shape
mlab_pca = mlabPCA(X)
print np.matrix(mlab_pca.Y[:,0]).shape
#transformed = np.matrix(mlab_pca.Y[:,0])
classA_train=[]
classB_train=[]
t_classA_train=[]
t_classB_train=[]
y_classA_train=[]
y_classB_train=[]
y_classA_test=[]
y_classB_test=[]
classA_test=[]
classB_test=[]
t_classA_test=[]
t_classB_test=[]

#print t_x_train
for i in range(x_train.shape[0]):
   
    if y_train[i] ==1:
        classA_train.append(X[i])
        #print t_x_train[i][0
        t_classA_train.append(t_x_train[i])
    else:
        classB_train.append(X[i])
        t_classB_train.append(t_x_train[i])

for i in range(x_test.shape[0]):
   
    if y_test[i] ==1:
        classA_test.append(X[i])
        t_classA_test.append(t_x_test[i])
    else:
        classB_test.append(X[i])
        t_classB_test.append(t_x_test[i])      
classA_train= np.array(np.matrix(classA_train).T)
classB_train= np.array(np.matrix(classB_train).T)
t_classA_train= np.array(t_classA_train)
t_classA_test= np.array(t_classA_test)
t_classB_test= np.array(t_classB_test)
t_classB_train= np.array(t_classB_train)


print t_classA_train.shape
print t_classA_test.shape

yclassB_predict = regClassify(t_classB_train,t_classB_test,np.zeros(len(t_classB_train)),np.zeros(len(t_classB_test)))
evaluate(np.zeros(len(t_classB_test)),yclassB_predict)
"""
"""
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
"""
