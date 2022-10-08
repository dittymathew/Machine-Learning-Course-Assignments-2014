import numpy as np

from sklearn.cross_validation import train_test_split




meanA=[1,0,0,0,0,0,0,0,0,0]

c =np.random.rand(10,10)
cov = c* np.transpose(c)
print cov
#cov =[[10,0,0,0,0,0,0,0,0,0],[0,9,0,0,0,0,0,0,0,0],[0,0,8,0,0,0,0,0,0,0],[0,0,0,7,0,0,0,0,0,0],[0,0,0,0,6,0,0,0,0,0],[0,0,0,0,0,5,0,0,0,0],[0,0,0,0,0,0,4,0,0,0],[0,0,0,0,0,0,0,3,0,0],[0,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,0,0,1]]
X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = np.random.multivariate_normal(meanA,cov,1000).T

y=[]
X=[]
i=0
for i in range(0,1000):
  y.append([0])
  X.append([X1[i],X2[i],X3[i],X4[i],X5[i],X6[i],X7[i],X8[i],X9[i],X10[i],0])

meanB=[2,0,0,0,0,0,0,0,0,0]
X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = np.random.multivariate_normal(meanB,cov,1000).T

for i in range(0,1000):
  y.append([1])
  X.append([X1[i],X2[i],X3[i],X4[i],X5[i],X6[i],X7[i],X8[i],X9[i],X10[i],1])
X=np.matrix(X)
y=np.matrix(y)
X_train, X_test = train_test_split(X,test_size=0.4, random_state=0)
np.savetxt("Data/DS1-train.csv", X_train, delimiter=",")
np.savetxt("Data/DS1-test.csv", X_test, delimiter=",")
#np.savetxt("DS1/train_label.csv", Y_train, delimiter=",")
#np.savetxt("DS1/test_label.csv", Y_test, delimiter=",")
"""
test =open('DS1/test.csv','w')
test_lbl =open('DS1/test_label.csv','w')
#train =open('DS1/train.csv','w')
train_lbl =open('DS1/train_label.csv','w')
#for x_row in X_train:
#  x_str =''
#  for x in x_row:
#    x_str += str(x)+','
#  train.write(x_str[0:(len(x_str)-1)]+'\n')
#  writetoCsv('DS1/train.csv',x_str[0:(len(x_str)-1)]+'\n')
for y_row in Y_train:
  y_str =''
  for y in y_row:
    y_str += str(y)+','
  train_lbl.write(y_str[0:(len(y_str)-1)]+'\n')
for x_row in X_test:
  x_str =''
  for x in x_row:
    x_str += str(x)+','
  test.write(x_str[0:(len(x_str)-1)]+'\n')
for y_row in Y_test:
  y_str =''
  for y in y_row:
    y_str += str(y)+','
  test_lbl.write(y_str[0:(len(y_str)-1)]+'\n')
"""
