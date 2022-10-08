import math
import theano
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import numpy as np
import random
from sklearn.preprocessing import normalize

def sigmoid(x):
#  s =1.0/(1+np.exp(-1*x))
  s =np.tanh(x)
  return s

def dsigmoid(x):
  return np.array(sigmoid(x))*np.array(1-sigmoid(x))
#  return 1-(np.array(sigmoid(x))*np.array(sigmoid(x)))

def softmax(w):
  t=1.0
  e = np.exp(np.array(w) / t)
  dist = e / np.sum(e)
  return dist
     
def dsoftmax(x):
  return np.array(softmax(x))*np.array(1-softmax(x))

def weight(n_in ,n_out):
  W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
  return W_values

label ={1:[0,0,0,1],2:[0,0,1,0],3:[0,1,0,0],4:[1,0,0,0]}
noHiddenNodes =20
yeta_b = 0.00001
yeta_a = 0.00001
yeta =0.01
data =np.loadtxt(open("Dataset/DS2.csv","rb"), delimiter=",")
np.random.shuffle(data)
X_train=data[:,0:96]
#n_X_train= np.true_divide((X_train-np.amin(X_train,axis=0)),(np.amax(X_train,axis=0) -np.amin(X_train,axis=0)))
n_X_train = normalize(X_train, norm='l2', axis=1)
testdata =np.loadtxt(open("Dataset/DS2_test.csv","rb"), delimiter=",")
np.random.shuffle(testdata)
X_test=testdata[:,0:96]
#n_X_test= np.true_divide((X_test-np.amin(X_test,axis=0)),(np.amax(X_test,axis=0) -np.amin(X_test,axis=0)))
n_X_test = normalize(X_test, norm='l2', axis=1)
print data.shape
X= np.column_stack((np.ones(1006),n_X_train))
print X.shape
y=data[:,96]
#a = (np.random.rand(97,noHiddenNodes)-0.5)/50 # initial weight for alpha
a =weight(97,noHiddenNodes)
b =weight(noHiddenNodes+1,4)
#b =(np.random.rand(noHiddenNodes+1,4)-0.5)/50 #initial weight for beta
#for i in range(0,noHiddenNodes):
#  a[:,i] =a[:,i]/100000*(10**i)
#for i in range(0,4):
#  b[:,i] =b[:,i]/1009**i
#a =a*0.00106
#b=b*0.0106
np.random.shuffle(a)
np.random.shuffle(b)
improvement=1
print 'yeta a =',yeta_a ,' yeta b = ',yeta_b, 'hidden nodes = ',noHiddenNodes
#while improvement==1:
for i in range(0,50):
  alpha =np.zeros((97,noHiddenNodes))
  beta =np.zeros((noHiddenNodes+1,4))
  
  sum_sqerror_N =0
  data= np.column_stack((X,y))
  np.random.shuffle(data)
  X= data[:,0:97]
  y= data[:,97]
  print X.shape,y.shape
  for i in range(0,1006):
    Xi =X[i,:]
#    print np.matrix(a).shape,np.matrix(Xi).shape
   # print Xi
    Zi =sigmoid(np.matrix(a).T*np.matrix(Xi).T) #1xm

 #   print Zi
  #  print np.argmax(Zi)
    Zi= np.row_stack((np.ones(1),Zi))
 #   print Zi.shape
    
    Yi =np.matrix(b).T*np.matrix(Zi) # 1x4
  #  print np.argmin(Yi)
    Yi= softmax(Yi).T
#    Ymax =np.argmax(Yi)+1
#    print Yi,Ymax
#    Yi = label[Ymax]
#    Yi=Yi.T
#    print Yi
    error = (np.matrix(label[y[i]]) - Yi) # 1x4
#    print i,',',np.sum(error)
#    print  np.matrix(label[y[i]]),Yi
    sum_sq_error = np.sum(np.array(error)*np.array(error)) # 1x4
    Zb =(np.matrix(b).T*Zi)
    delta_ki = -2*error ###
#    print error.shape
#    delta_ki = -2*error.sum(axis=1) ###
    b_delta_km = (np.array(b)* np.array(delta_ki)).sum(axis=1)
    s_mi =dsigmoid(np.matrix(Xi)*np.matrix(a))*np.matrix(b_delta_km[0:noHiddenNodes]).T
    for p in range (1,97):
      for m in range(0,noHiddenNodes):
        
        alpha[p,m] = alpha[p,m] + s_mi*Xi[p]
    for m in range(0,noHiddenNodes):
      for k in range(0,4):
        beta[m,k] =beta[m,k] + delta_ki[:,k]*Zi[m,:]
    sum_sqerror_N += sum_sq_error
  #print alpha
 # print beta
  a=a-yeta_a*np.array(alpha)
#  print a
  b=b-yeta_b*np.array(beta)
#  adiff =sum(sum((a-anew)**2))
 # bdiff =sum(sum((b-bnew)**2))
  
#print adiff+bdiff,', ',yeta_b*100
 # if adiff+bdiff< yeta_b*1000:
  #  improvement =0
  #else:
   # improvement =1
  print sum_sqerror_N

  XTest= np.column_stack((np.ones(80),n_X_test))
  y_test=testdata[:,96]
  ZTest =sigmoid(np.matrix(XTest)*np.matrix(a)) #80xm
  ZTest= np.column_stack((np.ones(80),ZTest))

  YP =np.dot(np.matrix(ZTest),np.matrix(b)) #80x4
  YP= softmax(YP)
  y_predict=[]
  for i in range(80):
#    print YP[i,:]
    Ymax =np.argmax(YP[i,:])+1
    y_predict.append(Ymax)
  print 'Accuracy:' ,accuracy_score(y_test,y_predict)
  print confusion_matrix(y_test,y_predict)

print y_predict
print y_test
print confusion_matrix(y_test,y_predict)
print classification_report(y_test,y_predict)
print 'yeta a =',yeta_a ,' yeta b = ',yeta_b, 'hidden nodes = ',noHiddenNodes
print 'Accuracy:' ,accuracy_score(y_test,y_predict)





