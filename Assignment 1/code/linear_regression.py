# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 11:36:40 2014

@author: ditty
"""
import numpy as np
from sklearn import linear_model
from numpy import linalg
import matplotlib.pyplot as plt

def regression(train,test):
    x_train = train[:,0:121]
    y_train = train[:,122]
    x_test = test[:,0:121]
    y_test = test[:,122]
#    x_train = train[:,0:126]
#    y_train = train[:,126]
#    x_test = test[:,0:126]
#    y_test = test[:,126]
    X =np.c_[np.ones(x_train.shape[0]),np.matrix(x_train)]
    Y =np.matrix(y_train)
#    print X.shape,Y.shape
    beta = linalg.inv(X.transpose()*X)*(X.transpose()*Y.transpose())
    #print beta

    rss =0
    for i in range(0,len(x_test)):
        predict = np.c_[np.matrix(np.ones(1)),np.matrix(x_test[i])]*beta

        rss +=(predict - y_test[i])**2
 
    print 'Residual Error:',rss
    return rss

def ridgeRegression(train,test,lamda):
    x_train = train[:,0:121]
    y_train = train[:,122]
    x_test = test[:,0:121]
    y_test = test[:,122]
    regr = linear_model.Ridge(lamda)
    regr.fit(x_train,y_train)
    print 'Coefficients: ',regr.coef_ 
    rss =0
    for i in range(0,len(x_test)):
        rss +=(regr.predict(x_test[i]) - y_test[i])**2
    return rss

def avgRssRidge(train1,train2,train3,train4,train5,test1,test2,test3,test4,test5,lamda):

    sumrss = ridgeRegression(train1,test1,lamda) + ridgeRegression(train2,test2,lamda) + ridgeRegression(train3,test3,lamda) +ridgeRegression(train4,test4,lamda) +ridgeRegression(train5,test5,lamda)
    avgrss = float(sumrss)/5
    return avgrss
    
    

d1_train =np.loadtxt(open("Data/CandC-train1.csv","rb"), delimiter = ',')
d1_test =np.loadtxt(open("Data/CandC-test1.csv","rb"), delimiter = ',')
d2_train =np.loadtxt(open("Data/CandC-train2.csv","rb"), delimiter = ',')
d2_test =np.loadtxt(open("Data/CandC-test2.csv","rb"), delimiter = ',')
d3_train =np.loadtxt(open("Data/CandC-train3.csv","rb"), delimiter = ',')
d3_test =np.loadtxt(open("Data/CandC-test3.csv","rb"), delimiter = ',')
d4_train =np.loadtxt(open("Data/CandC-train4.csv","rb"), delimiter = ',')
d4_test =np.loadtxt(open("Data/CandC-test4.csv","rb"), delimiter = ',')
d5_train =np.loadtxt(open("Data/CandC-train5.csv","rb"), delimiter = ',')
d5_test =np.loadtxt(open("Data/CandC-test5.csv","rb"), delimiter = ',')


sum_err =0
error =regression(d1_train,d1_test)
sum_err +=error
min_error= error
best_data ='CandC-train1.csv ,Cand-test1.csv'

error =regression(d2_train,d2_test)
sum_err +=error
if error<min_error :
    min_error= error
    best_data ='CandC-train2.csv ,Cand-test2.csv'
error =regression(d3_train,d3_test)
sum_err +=error
if error<min_error :
    min_error= error
    best_data ='CandC-train3.csv ,Cand-test3.csv'
error =regression(d4_train,d4_test)
sum_err +=error
if error<min_error :
    min_error= error
    best_data ='CandC-train4.csv ,Cand-test4.csv'
error =regression(d5_train,d5_test)
sum_err +=error

if error<min_error :
    min_error= error
    best_data ='CandC-train5.csv ,Cand-test5.csv'

print min_error
print best_data
avg_err = float(sum_err)/float(5)
print 'Average Error=',avg_err
lamda =1
e=avgRssRidge(d1_train,d2_train,d3_train,d4_train,d5_train,d1_test,d2_test,d3_test,d4_test,d5_test,lamda)
"""
i =0
err=[]
l=[]
k=1
while(i<=100):
  lamda = i
  k=k*10
  l.append(lamda)
  e=avgRssRidge(d1_train,d2_train,d3_train,d4_train,d5_train,d1_test,d2_test,d3_test,d4_test,d5_test,lamda)
  print lamda, e
  err.append(e)
  i +=10
fig =plt.figure()
plt.plot(l,err,marker ='.')
plt.xlabel('lamda')
plt.ylabel('error')
plt.show()
"""
