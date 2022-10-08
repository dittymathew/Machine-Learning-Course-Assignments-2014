import math
import random
import string
import Image
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

from sklearn.cross_validation import KFold

training_data = np.loadtxt(open('DS2-train.csv',"rb"),delimiter=",")
test_data = np.loadtxt(open('DS2-test.csv',"rb"),delimiter=",")

#shuffle the data
np.random.shuffle(training_data);

Y_train_num = training_data[:,training_data.shape[1]-1]
X_train = training_data[:,0:training_data.shape[1]-1]

Y_test_num = test_data[:,training_data.shape[1]-1]
X_test = test_data[:,0:training_data.shape[1]-1]

labels = {0:[1,0,0,0], 1:[0,1,0,0], 2:[0,0,1,0], 3:[0,0,0,1]}

Y_train = np.empty((0,4))
Y_test = np.empty((0,4))

for i in range(0,Y_train_num.shape[0]):
	Y_train = np.concatenate((Y_train, [labels[Y_train_num[i]]]), axis=0)

for i in range(0,Y_test_num.shape[0]):
	Y_test = np.concatenate((Y_test, [labels[Y_test_num[i]]]), axis=0)	

#normalize the test and train
X_norm = normalize(X_train, norm='l2', axis=1)

X_test_norm = normalize(X_test, norm='l2', axis=1)

#############################################  data is processed   ###################################################################

no_input_units = 97 # 96 + 1 bias
no_hidden_units = 101 # 100 + 1 bias
no_output_units = 4 

Z = np.zeros(no_hidden_units) # vectors
X = np.zeros(no_input_units)
Y = np.zeros(no_output_units) 


def sigmoid (x):
  return (1.0) / (1 + np.exp(-x))
  
def dsigmoid (y):
  return sigmoid(y)*(1-sigmoid(y))

def softmax (T):
	#return (1.0) / (1 + np.exp(-T)) #sigmoid
  return np.exp(T) / np.sum(np.exp(T)) #softmax

def dsoftmax (y):
  return softmax(y)*(1-softmax(y))

def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = np.random.uniform(a,b,1)

def backPropagate (targets, N, M, regularization, lambdap):
	global X,Y,Z,alpha,beta
	output_deltas = [0.0] * no_output_units
	error = targets - Y
	output_deltas = 2 *  error * dsoftmax(np.dot(Z.T, beta)) 

	for j in range(no_hidden_units):
		for k in range(no_output_units):
			change = output_deltas[k] * Z[j]
			if (regularization == 1):
				change += lambdap*2*beta[j][k]
			beta[j][k] += N*change 

	hidden_deltas = [0.0] * no_hidden_units
	for j in range(no_hidden_units):
		error = 0.0
		for k in range(no_output_units):
			error += output_deltas[k] * beta[j][k]
		hidden_deltas[j] = error * dsigmoid(np.dot(X.T ,alpha[:,j]))
	
	for i in range (no_input_units):
		for j in range (no_hidden_units):
			if j != 0:
				change = hidden_deltas[j] * X[i]
				if (regularization == 1):
					change += lambdap*2*alpha[i][j]
				alpha[i][j] += M*change    

	# calc combined error
	# 1/2 for differential convenience & **2 for modulus
	error = 0.0
	for k in range(len(targets)):
		error = 0.5 * (targets[k]-Y[k])**2
	return error
        
def fit (regularization, lambdap, max_iterations = 25, N=0.001, M=0.01):
	global X,Y,Z,alpha,beta
	T = np.zeros(no_output_units) 
	error = 0.0
	for itr in range(max_iterations):
		for i in range(0,X_input.shape[0]):
			# forward pass
			X = X_input[i,:]
			for j in range(no_hidden_units):
			      if  j == 0:
				      Z[j] = 1
			      else:	
				      sum_ =np.dot((X_input[i,:]).T ,alpha[:,j])
				      Z[j] = sigmoid (sum_) 
			for k in range(no_output_units):
			       T[k] =np.dot(Z.T, beta[:,k])
			Y = softmax (T) 
			# backward pass
			error = backPropagate(Y_train[i], N, M, regularization, lambdap)
		print 'Combined error', error
        print 'done'
    
def predict(X_test_in):
	global X,Y,Z,alpha,beta
	Y_pred = np.empty((0,4))
	ones = np.array([np.ones(X_test_in.shape[0],)])
	X_testt =  np.concatenate((ones.T, X_test_in),axis=1)
	T = np.zeros(no_output_units) 
	for i in range(0,X_testt.shape[0]):
		# forward pass
		for j in range(no_hidden_units):
		      if  j == 0:
			      Z[j] = 1
		      else:	
			      sum_ =np.dot((X_testt[i,:]).T ,alpha[:,j])
			      Z[j] = sigmoid (sum_) 
		for k in range(no_output_units):
		       T[k] =np.dot(Z.T, beta[:,k])
		Y = softmax (T)  
		Y_pred = np.concatenate((Y_pred, [Y]), axis=0)
	for i in range(0, Y_pred.shape[0]):
		max = -1.0
		label = -1
		for j in range(0, Y_pred.shape[1]):
			if(Y_pred[i][j] > max):
				max = Y_pred[i][j] 
				label = j;
		Y_pred[i] =  labels[label]
	return Y_pred

alpha = np.empty((no_input_units , no_hidden_units))   # matrix : no_hidden_units * no_input_units 
beta =  np.empty((no_hidden_units, no_output_units))   # matrix : no_output_units * no_hidden_units
   
# initialize node weights to random vals
randomizeMatrix ( alpha, -0.5, 0.5)
randomizeMatrix ( beta, -1, 1  )


ones = np.array([np.ones(X_norm.shape[0],)])
X_input =  np.concatenate((ones.T, X_norm),axis=1)


fit(0, 0.0)
y_train_pred = predict(X_norm)
y_test_pred = predict(X_test_norm)
target_names = ['coast', 'forest', 'insidecity', 'mountain']
print(classification_report(Y_train, y_train_pred, target_names=target_names))
print(classification_report(Y_test, y_test_pred, target_names=target_names))


