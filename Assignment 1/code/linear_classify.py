import numpy as np
from numpy import linalg
#from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def knnClassify(X_train,X_test,Y_train,Y_test):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, Y_train)
  pclassA=[]
  pclassB=[]
  Y_predict =[]
  for i in range(0,800):
    if knn.predict(X_test[i]) ==0:
   
      pclassA.append(list(X_test[i]))
      Y_predict.append(0)
    else:
      pclassB.append(list(X_test[i]))
      Y_predict.append(1)
  return Y_predict

def regClassify(X_train,X_test,Y_train,Y_test):
#  regr = linear_model.LinearRegression()
#  regr.fit(X_train,Y_train)
  
  X =np.c_[np.ones(X_train.shape[0]),np.matrix(X_train)]
  Y =np.matrix(Y_train)
  print X.shape,Y.shape
  beta = linalg.inv(X.transpose()*X)*(X.transpose()*Y.transpose())
  print beta
#  print 'Coefficient =',regr.coef_
  pclassA=[]
  pclassB=[]
  Y_predict =[]
  for i in range(0,800):
    
    predict = np.c_[np.matrix(np.ones(1)),np.matrix(X_test[i])]*beta
    #if regr.predict(X_test[i]) <=0.5:
    if predict <= 0.5:
      pclassA.append(list(X_test[i]))
      Y_predict.append(0)
    else:
      pclassB.append(list(X_test[i]))
      Y_predict.append(1)
  return Y_predict

def evaluate(Y_test,Y_predict):
  acc =metrics.accuracy_score(Y_test,Y_predict)
  precision =metrics.precision_score(Y_test,Y_predict)
  recall =metrics.recall_score(Y_test,Y_predict)
  fmeasure =metrics.f1_score(Y_test,Y_predict)
  print '****************************************************************'
  print 'Confusion Matrix :',metrics.confusion_matrix(Y_test,Y_predict)
  print 'Accuracy =',acc
  print 'Precision =',precision
  print 'Recall =',recall
  print 'F-measure =',fmeasure

def getData(f1,f2):
  X1 =  np.loadtxt(open(f1,"rb"), delimiter = ',')
  X2 =  np.loadtxt(open(f2,"rb"), delimiter = ',')
#  Y_train =  np.loadtxt(open(f3,"rb"), delimiter = ',')
 # Y_test =  np.loadtxt(open(f4,"rb"), delimiter = ',')
  Y_train = X1[:,10]
  Y_test = X2[:,10]
  X_train = X1[:,0:10]
  X_test = X2[:,0:10]
  print X_test.shape,Y_test.shape
  return X_train,X_test,Y_train,Y_test

k=300
print "k =",k
print "Single Multivariate Gaussian Distribution per class"

x_train,x_test,y_train,y_test = getData("Data/DS1-train.csv","Data/DS1-test.csv")
y_predict = regClassify(x_train,x_test,y_train,y_test)
print "Linear Classifier using regression"
evaluate(y_test,y_predict)
y_predict = knnClassify(x_train,x_test,y_train,y_test)
print "Knn Classifier"
evaluate(y_test,y_predict)

print "Mixture of 3 Gaussians"
x_train,x_test,y_train,y_test = getData("Data/DS2-train.csv","Data/DS2-test.csv")
y_predict = regClassify(x_train,x_test,y_train,y_test)
print "Linear Classifier using regression"
evaluate(y_test,y_predict)
y_predict = knnClassify(x_train,x_test,y_train,y_test)
print "Knn Classifier"
evaluate(y_test,y_predict)
