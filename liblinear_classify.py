import numpy as np
from sklearn.preprocessing import normalize
from sklearn import  linear_model
from sklearn.metrics import classification_report
import os
from sklearn import metrics
from liblinearutil import *

#3500 * 1897
training_data = np.loadtxt(open('data/contest-train1_ip.csv',"rb"),delimiter=",")
test_data = np.loadtxt(open('data/contest-test1_ip.csv',"rb"),delimiter=",")

Y_train = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

Y_test = test_data[:,training_data.shape[1]-1]
X_test = test_data[:,0:training_data.shape[1]-1]

X_norm = normalize(X_train, norm='l2', axis=1)
X_test = normalize(X_test, norm='l2', axis=1)

#############################    Data is processed ############################################

prob  = problem(Y_train, X_norm)
param = parameter('-s 0 -c 4 -B 1')
m = train(prob, param)
save_model('classify.model', m)
m = load_model('classify.model')
p_label, p_acc, p_val = predict(Y_test, X_test, m, '-b 1')
ACC, MSE, SCC = evaluations(y, p_label)
print ACC
print MSE
print SCC


