from PIL import Image
import joblib
#from sklearn.externals import joblib
import pickle
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import cross_validation

category ={'coast':1,'forest':2,'insidecity':3,'mountain':4}
train =np.loadtxt(open("DS2-train.csv","rb"), delimiter=",")
test = np.loadtxt(open("DS2-test.csv","rb"), delimiter=",")
print len(train)
X_train =train[:,0:96]
X_test =test[:,0:96]
y_test=test[:,96]
y_train=train[:,96]
n_X_train= np.true_divide((X_train-np.amin(X_train,axis=0)),(np.amax(X_train,axis=0) -np.amin(X_train,axis=0)))
n_X_test= np.true_divide((X_test-np.amin(X_test,axis=0)),(np.amax(X_test,axis=0) -np.amin(X_test,axis=0)))
i=0
while i<1006:
  if y_train[i] == 1. : 
    coast_train= n_X_train[i:i+265,:]
    i=i+265
  if y_train[i] == 2. : 
    forest_train=n_X_train[i:i+239,:]
    i=i+239
  if y_train[i] == 3. : 
    insidecity_train =n_X_train[i:i+223,:]
    i=i+223
  if y_train[i] == 4. : 
    mountain_train =n_X_train[i:i+279,:]
    i=i+279
print coast_train.shape,forest_train.shape,insidecity_train.shape,mountain_train.shape
  

coast_train =np.column_stack((np.array(coast_train),np.ones(265)))
forest_train =np.column_stack((np.array(forest_train),np.ones(239)*2))
insidecity_train =np.column_stack((np.array(insidecity_train),np.ones(223)*3))
mountain_train =np.column_stack((np.array(mountain_train),np.ones(279)*4))

##coast cross validation
coast_kfold =cross_validation.KFold(n=265,n_folds=5)
coast_train_cv=[]
coast_test_cv=[]
for train,test in coast_kfold:
  coast_train_cv.append(coast_train[train,:])
  coast_test_cv.append(coast_train[test,:])


##forest cross validation
forest_kfold =cross_validation.KFold(n=239,n_folds=5)
forest_train_cv=[]
forest_test_cv=[]
for train,test in forest_kfold:
  forest_train_cv.append(forest_train[train,:])
  forest_test_cv.append(forest_train[test,:])


##insidecity cross validation
insidecity_kfold =cross_validation.KFold(n=223,n_folds=5)
insidecity_train_cv=[]
insidecity_test_cv=[]
for train,test in insidecity_kfold:
  insidecity_train_cv.append(insidecity_train[train,:])
  insidecity_test_cv.append(insidecity_train[test,:])


##mountain cross validation
mountain_kfold =cross_validation.KFold(n=265,n_folds=5)
mountain_train_cv=[]
mountain_test_cv=[]
for train,test in mountain_kfold:
  mountain_train_cv.append(mountain_train[train,:])
  mountain_test_cv.append(mountain_train[test,:])

cv1_train =np.concatenate((np.array(coast_train_cv[0]),np.array(forest_train_cv[0]),np.array(insidecity_train_cv[0]),np.array(mountain_train_cv[0])))
cv1_test =np.concatenate((np.array(coast_test_cv[0]),np.array(forest_test_cv[0]),np.array(insidecity_test_cv[0]),np.array(mountain_test_cv[0])))
cv2_train =np.concatenate((np.array(coast_train_cv[1]),np.array(forest_train_cv[1]),np.array(insidecity_train_cv[1]),np.array(mountain_train_cv[1])))
cv2_test =np.concatenate((np.array(coast_test_cv[1]),np.array(forest_test_cv[1]),np.array(insidecity_test_cv[1]),np.array(mountain_test_cv[1])))
cv3_train =np.concatenate((np.array(coast_train_cv[2]),np.array(forest_train_cv[2]),np.array(insidecity_train_cv[2]),np.array(mountain_train_cv[2])))
cv3_test =np.concatenate((np.array(coast_test_cv[2]),np.array(forest_test_cv[2]),np.array(insidecity_test_cv[2]),np.array(mountain_test_cv[2])))
cv4_train =np.concatenate((np.array(coast_train_cv[3]),np.array(forest_train_cv[3]),np.array(insidecity_train_cv[3]),np.array(mountain_train_cv[3])))
cv4_test =np.concatenate((np.array(coast_test_cv[3]),np.array(forest_test_cv[3]),np.array(insidecity_test_cv[3]),np.array(mountain_test_cv[3])))
cv5_train =np.concatenate((np.array(coast_train_cv[4]),np.array(forest_train_cv[4]),np.array(insidecity_train_cv[4]),np.array(mountain_train_cv[4])))
cv5_test =np.concatenate((np.array(coast_test_cv[4]),np.array(forest_test_cv[4]),np.array(insidecity_test_cv[4]),np.array(mountain_test_cv[4])))


##Linear Kernel  
best=0
for c in [1,100,1000,10000]:
  sum_acc =0
  #print "***************************************************************"
  #print "C= ",c
  #print "***************************************************************"
  classifier = svm.SVC(kernel='linear',C=c)
  y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
  acc = accuracy_score(cv1_test[:,96], y_pred)
#  print "Linear Kernel, 1, Accuracy = ",acc
  sum_acc +=acc

  y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
  acc = accuracy_score(cv2_test[:,96], y_pred)
#  print "Linear Kernel, 2, Accuracy = ",acc
  sum_acc +=acc

  y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
  acc = accuracy_score(cv3_test[:,96], y_pred)
#  print "Linear Kernel, 3, Accuracy = ",acc
  sum_acc +=acc

  y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
  acc = accuracy_score(cv4_test[:,96], y_pred)
 # print "Linear Kernel, 4, Accuracy = ",acc
  sum_acc +=acc

  y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
  acc = accuracy_score(cv5_test[:,96], y_pred)
  #print "Linear Kernel, 5, Accuracy = ",acc
  sum_acc +=acc
  avg_acc= float(sum_acc)/float(5)
  print 'C= ',c,' Acc = ',avg_acc
  if avg_acc>best:
    best =avg_acc
    best_c =c
print "Best c", best_c
classifier = svm.SVC(kernel='linear',C=best_c)
"""
y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
acc = accuracy_score(cv1_test[:,96], y_pred)
print "Linear Kernel, 1, Accuracy = ",acc

y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
acc = accuracy_score(cv2_test[:,96], y_pred)
print "Linear Kernel, 2, Accuracy = ",acc

y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
acc = accuracy_score(cv3_test[:,96], y_pred)
print "Linear Kernel, 3, Accuracy = ",acc

y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
acc = accuracy_score(cv4_test[:,96], y_pred)
print "Linear Kernel, 4, Accuracy = ",acc

y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
acc = accuracy_score(cv5_test[:,96], y_pred)
print "Linear Kernel, 5, Accuracy = ",acc
"""

print "***************************************************************"
classifier.fit(n_X_train, y_train)
joblib.dump(classifier, 'Model/CS13D018_linear.pkl')
y_pred =classifier.fit(n_X_train,y_train).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Linear Kernel, Test set, Accuracy = ",acc



print "***************************************************************"
##Polynomial Kernel  
best =0
for c in [1,100,1000,10000]:
  for d in [1,2,3]:
    sum_acc=0
 #   print "***************************************************************"
 #   print "C= ",c," degree= ",d
 #   print "***************************************************************"
    classifier = svm.SVC(kernel='poly',C=c,degree=d)
    y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
    acc = accuracy_score(cv1_test[:,96], y_pred)
   # print "Polynomial Kernel, 1, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
    acc = accuracy_score(cv2_test[:,96], y_pred)
    #print "Polynomial Kernel, 2, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
    acc = accuracy_score(cv3_test[:,96], y_pred)
   # print "Polynomial Kernel, 3, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
    acc = accuracy_score(cv4_test[:,96], y_pred)
    #print "Polynomial Kernel, 4, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
    acc = accuracy_score(cv5_test[:,96], y_pred)
   # print "Polynomial Kernel, 5, Accuracy = ",acc
    sum_acc +=acc
    avg_acc =float(sum_acc)/float(5)
    print 'C= ',c,'D= ',d,'Acc = ', avg_acc
    if avg_acc>best:
      best =avg_acc
      best_c =c
      best_d=d
print "***************************************************************"
print 'best c ',best_c,' best_d ',best_d
classifier.fit(n_X_train, y_train)
joblib.dump(classifier, 'Model/CS13D018_polynomial.pkl')
y_pred =classifier.fit(n_X_train,y_train).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Polynomial Kernel, Test set, Accuracy = ",acc
"""
classifier = svm.SVC(kernel='poly',C=best_c,degree=best_d)
y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
acc = accuracy_score(cv1_test[:,96], y_pred)
print "Polynomial Kernel, 1, Accuracy = ",acc

y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
acc = accuracy_score(cv2_test[:,96], y_pred)
print "Polynomial Kernel, 2, Accuracy = ",acc

y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
acc = accuracy_score(cv3_test[:,96], y_pred)
print "Polynomial Kernel, 3, Accuracy = ",acc

y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
acc = accuracy_score(cv4_test[:,96], y_pred)
print "Polynomial Kernel, 4, Accuracy = ",acc

y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
acc = accuracy_score(cv5_test[:,96], y_pred)
print "Polynomial Kernel, 5, Accuracy = ",acc
print "***************************************************************"

classifier.fit(cv5_train[:,0:96], cv5_train[:,96])
joblib.dump(classifier, 'Model/CS13D018_polynomial.pkl')
y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Polynomial Kernel, Test set using 5, Accuracy = ",acc
"""
print "***************************************************************"
##Gaussian Kernel  
best =0
for c in [1,100,1000,10000]:
  for g in [0.1,0.01,0.001,1,10,100]:
    sum_acc=0
#    print "***************************************************************"
#    print "C= ",c," Gamma= ",g
#    print "***************************************************************"
    classifier = svm.SVC(kernel='rbf',C=c,gamma=g)
    y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
    acc = accuracy_score(cv1_test[:,96], y_pred)
#    print "Gaussian Kernel, 1, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
    acc = accuracy_score(cv2_test[:,96], y_pred)
 #   print "Gaussian Kernel, 2, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
    acc = accuracy_score(cv3_test[:,96], y_pred)
  #  print "Gaussian Kernel, 3, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
    acc = accuracy_score(cv4_test[:,96], y_pred)
 #   print "Gaussian Kernel, 4, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
    acc = accuracy_score(cv5_test[:,96], y_pred)
  #  print "Gaussian Kernel, 5, Accuracy = ",acc
    sum_acc +=acc
    avg_acc =float(sum_acc)/float(5)
    print 'C= ',c,' G = ',g,' Acc= ',avg_acc
    if avg_acc>best:
      best =avg_acc
      best_c =c
      best_g=g

print "***************************************************************"
print 'best c ',best_c,' best gamma ',best_g
classifier.fit(n_X_train, y_train)
joblib.dump(classifier, 'Model/CS13D018_gaussian.pkl')
y_pred =classifier.fit(n_X_train,y_train).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Gaussian Kernel, Test set, Accuracy = ",acc
"""
classifier = svm.SVC(kernel='rbf',gamma=best_g,C=best_c)
y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
acc = accuracy_score(cv1_test[:,96], y_pred)
print "Gaussain Kernel, 1, Accuracy = ",acc

y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
acc = accuracy_score(cv2_test[:,96], y_pred)
print "Gaussian Kernel, 2, Accuracy = ",acc

y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
acc = accuracy_score(cv3_test[:,96], y_pred)
print "Gaussian Kernel, 3, Accuracy = ",acc

y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
acc = accuracy_score(cv4_test[:,96], y_pred)
print "Gaussian Kernel, 4, Accuracy = ",acc

y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
acc = accuracy_score(cv5_test[:,96], y_pred)
print "Gaussian Kernel, 5, Accuracy = ",acc
print "***************************************************************"

classifier.fit(cv5_train[:,0:96], cv5_train[:,96])
joblib.dump(classifier, 'Model/CS13D018_gaussian.pkl')
y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Gaussian Kernel, Test set using 5, Accuracy = ",acc
print "***************************************************************"
"""
##Sigmoid Kernel  
best=0
for c in [1,100,1000,10000]:
  for g in [0.01,0.1,1,10,100]:
    sum_acc =0
#  print "***************************************************************"
#  print "C= ",c
#  print "***************************************************************"
    classifier = svm.SVC(kernel='sigmoid',C=c,gamma=g)
    y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
    acc = accuracy_score(cv1_test[:,96], y_pred)
 # print "Sigmoid Kernel, 1, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
    acc = accuracy_score(cv2_test[:,96], y_pred)
  #print "Sigmoid Kernel, 2, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
    acc = accuracy_score(cv3_test[:,96], y_pred)
 # print "Sigmoid Kernel, 3, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
    acc = accuracy_score(cv4_test[:,96], y_pred)
    #print "Sigmoid Kernel, 4, Accuracy = ",acc
    sum_acc +=acc

    y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
    acc = accuracy_score(cv5_test[:,96], y_pred)
#  print "Sigmoid Kernel, 5, Accuracy = ",acc
    sum_acc +=acc
    avg_acc= float(sum_acc)/float(5)
    print 'C= ',c, ' Acc =',avg_acc
    if avg_acc>best:
      best =avg_acc
      best_c =c
      best_g =g
  
print "***************************************************************"
print 'best c ',best_c,' best_g ',best_g
classifier.fit(n_X_train, y_train)
joblib.dump(classifier, 'Model/CS13D018_sigmoid.pkl')
y_pred =classifier.fit(n_X_train,y_train).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Sigmoid Kernel, Test set, Accuracy = ",acc
"""
classifier = svm.SVC(kernel='sigmoid',C=best_c)
y_pred =classifier.fit(cv1_train[:,0:96], cv1_train[:,96]).predict(cv1_test[:,0:96])
acc = accuracy_score(cv1_test[:,96], y_pred)
print "Sigmoid Kernel, 1, Accuracy = ",acc

y_pred =classifier.fit(cv2_train[:,0:96], cv2_train[:,96]).predict(cv2_test[:,0:96])
acc = accuracy_score(cv2_test[:,96], y_pred)
print "Sigmoid Kernel, 2, Accuracy = ",acc

y_pred =classifier.fit(cv3_train[:,0:96], cv3_train[:,96]).predict(cv3_test[:,0:96])
acc = accuracy_score(cv3_test[:,96], y_pred)
print "Sigmoid Kernel, 3, Accuracy = ",acc

y_pred =classifier.fit(cv4_train[:,0:96], cv4_train[:,96]).predict(cv4_test[:,0:96])
acc = accuracy_score(cv4_test[:,96], y_pred)
print "Sigmoid Kernel, 4, Accuracy = ",acc

y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(cv5_test[:,0:96])
acc = accuracy_score(cv5_test[:,96], y_pred)
print "Sigmoid Kernel, 5, Accuracy = ",acc


print "***************************************************************"

classifier.fit(cv5_train[:,0:96], cv5_train[:,96])
joblib.dump(classifier, 'Model/CS13D018_sigmoid.pkl')
y_pred =classifier.fit(cv5_train[:,0:96], cv5_train[:,96]).predict(n_X_test[:,0:96])
acc = accuracy_score(y_test, y_pred)
print "Sigmoid Kernel, Test set using 5, Accuracy = ",acc
print "***************************************************************"
"""
