import Image
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.metrics import classification_report
import joblib

training_data = np.loadtxt(open('data/contest-train1.csv',"rb"),delimiter=",")
test_data = np.loadtxt(open('data/contest-test1.csv',"rb"),delimiter=",")

Y_train = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

Y_test = test_data[:,training_data.shape[1]-1]
X_test = test_data[:,0:training_data.shape[1]-1]


X_norm = normalize(X_train, norm='l2', axis=1)

X_test_norm = normalize(X_test, norm='l2', axis=1)


#come up with the parameters using n-fold crossvalidation


kf = KFold(X_norm.shape[0], n_folds=5, shuffle=True, random_state=32)
best_f1_score = 0.0
best_c = 1.0
for c_val in [1.0, 5.0, 10.0, 100.0, 1000.0]:
	sum_f1_score = 0.0
	ave_f1_score = 0.0
	sum_acc_score = 0.0
	ave_acc_score = 0.0
	for train, test in kf:
		X_cv_train, Y_cv_train = X_norm[train], Y_train[train]
		X_cv_test,  Y_cv_test = X_norm[test], Y_train[test]
		model1 = svm.SVC(C=c_val, kernel='linear')
		model1.fit(X_cv_train, Y_cv_train) 
		Y_cv_test_pred = model1.predict(X_cv_test)
		acc = metrics.accuracy_score(Y_cv_test, Y_cv_test_pred)
		f1_score = metrics.f1_score(Y_cv_test, Y_cv_test_pred)
		sum_f1_score +=  f1_score
		sum_acc_score +=  acc
	ave_f1_score = float(sum_f1_score)/float(5)
	ave_acc_score = float(sum_acc_score)/float(5)
	if (ave_f1_score > best_f1_score): 
		best_c = c_val	
		best_f1_score = ave_f1_score
	print  "average:", ave_acc_score, ave_f1_score, best_c
	


# put modelx.coeff_  for each model x in to CS13D201.mat
model1 = svm.SVC(C=best_c, kernel='linear')
model1.fit(X_norm, Y_train)	 #default 3 
Y_test_pred = model1.predict(X_test_norm)
target_names = ['coast', 'forest', 'insidecity', 'mountain']
print(classification_report(Y_test, Y_test_pred, target_names=target_names))

best_f1_score = 0.0
best_deg = 1
for deg in [1, 2, 3, 4, 5]: # c= 10000 deg=2
	sum_f1_score = 0.0
	ave_f1_score = 0.0
	sum_acc_score = 0.0
	ave_acc_score = 0.0
	for train, test in kf:
		X_cv_train, Y_cv_train = X_norm[train], Y_train[train]
		X_cv_test,  Y_cv_test = X_norm[test], Y_train[test]
		model2 = svm.SVC(C=5.0, kernel='poly', degree=deg) 
		model2.fit(X_cv_train, Y_cv_train) 
		Y_cv_test_pred = model2.predict(X_cv_test)
		#print Y_cv_test
		#print Y_cv_test_pred
		acc = metrics.accuracy_score(Y_cv_test, Y_cv_test_pred)
		f1_score = metrics.f1_score(Y_cv_test, Y_cv_test_pred)
		sum_f1_score +=  f1_score
		sum_acc_score +=  acc
	ave_f1_score = float(sum_f1_score)/float(5)
	ave_acc_score = float(sum_acc_score)/float(5)
	if (ave_f1_score > best_f1_score): 
		best_deg =deg	
		best_f1_score = ave_f1_score
	print "average:", ave_acc_score, ave_f1_score, best_deg
	

model2 = svm.SVC(C=0.1, kernel='poly', degree=best_deg) 
model2.fit(X_norm, Y_train)
Y_test_pred = model2.predict(X_test_norm)
target_names = ['coast', 'forest', 'insidecity', 'mountain']
print(classification_report(Y_test, Y_test_pred, target_names=target_names))

best_gam = 1.0
best_f1_score = 0.0
for gam in [0.001, 0.01, 0.1, 1, 2]: #C, 1000 c=1 
	sum_acc_score = 0.0
	ave_acc_score = 0.0
	sum_f1_score = 0.0
	ave_f1_score = 0.0
	for train, test in kf:
		X_cv_train, Y_cv_train = X_norm[train], Y_train[train]
		X_cv_test,  Y_cv_test = X_norm[test], Y_train[test]
		model3 = svm.SVC(C=1.0, kernel='rbf', gamma=gam) 
		model3.fit(X_cv_train, Y_cv_train) 
		Y_cv_test_pred = model3.predict(X_cv_test)
		acc = metrics.accuracy_score(Y_cv_test, Y_cv_test_pred)
		f1_score = metrics.f1_score(Y_cv_test, Y_cv_test_pred)
		sum_f1_score +=  f1_score
		sum_acc_score +=  acc
	ave_f1_score = float(sum_f1_score)/float(5)
	ave_acc_score = float(sum_acc_score)/float(5)
	if (ave_f1_score > best_f1_score): 
		best_gam  =gam	
		best_f1_score = ave_f1_score
	print "average:", ave_acc_score, ave_f1_score, best_gam 


model3 = svm.SVC(kernel='rbf', gamma=best_gam, degree=1)
model3.fit(X_norm, Y_train)
Y_test_pred = model3.predict(X_test_norm)
target_names = ['coast', 'forest', 'insidecity', 'mountain']
print(classification_report(Y_test, Y_test_pred, target_names=target_names))


best_f1_score = 0.0
best_c = 1.0
for c_val in [1.0, 5.0, 10.0, 100.0, 1000.0]:
	sum_acc_score = 0.0
	ave_acc_score = 0.0
	sum_f1_score = 0.0
	ave_f1_score = 0.0
	for train, test in kf:
		X_cv_train, Y_cv_train = X_norm[train], Y_train[train]
		X_cv_test,  Y_cv_test = X_norm[test], Y_train[test]
		model4 = svm.SVC(C=c_val, kernel='sigmoid')
		model4.fit(X_cv_train, Y_cv_train) 
		Y_cv_test_pred = model4.predict(X_cv_test)
		acc = metrics.accuracy_score(Y_cv_test, Y_cv_test_pred)
		f1_score = metrics.f1_score(Y_cv_test, Y_cv_test_pred)
		sum_f1_score +=  f1_score
		sum_acc_score +=  acc
	ave_f1_score = float(sum_f1_score)/float(5)
	ave_acc_score = float(sum_acc_score)/float(5)
	if (ave_f1_score > best_f1_score): 
		best_c = c_val	
		best_f1_score = ave_f1_score
	print "average:", ave_acc_score, ave_f1_score, best_c
	
model4 = svm.SVC(C=best_c, kernel='sigmoid')  
model4.fit(X_norm, Y_train)
Y_test_pred = model4.predict(X_test_norm)
target_names = ['coast', 'forest', 'insidecity', 'mountain']
print(classification_report(Y_test, Y_test_pred, target_names=target_names))


