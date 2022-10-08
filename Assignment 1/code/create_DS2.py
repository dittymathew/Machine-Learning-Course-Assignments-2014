import numpy as np

from sklearn.cross_validation import train_test_split




mean1A=[1,1,1,1,1,1,1,1,1,1]
mean2A=[5,5,5,5,5,5,5,5,5,5]
mean3A=[10,10,10,10,10,10,10,10,10,10]
mean1B=[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
mean2B=[5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]
mean3B=[10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5]
#o =np.ones((10))
#mean1A =np.random.rand(10)
#mean1B = mean1A + o
#mean2A =np.random.rand(10)
#mean2B = mean2A + o
#mean3A =np.random.rand(10)
#mean3B = mean3A + o
c =np.random.rand(10,10)
cov1 = c* np.transpose(c)
c =np.random.rand(10,10)
cov2 = c* np.transpose(c)
c =np.random.rand(10,10)
cov3 = c* np.transpose(c)
#cov1 =[[10,0,0,0,0,0,0,0,0,0],[0,9,0,0,0,0,0,0,0,0],[0,0,8,0,0,0,0,0,0,0],[0,0,0,7,0,0,0,0,0,0],[0,0,0,0,6,0,0,0,0,0],[0,0,0,0,0,5,0,0,0,0],[0,0,0,0,0,0,4,0,0,0],[0,0,0,0,0,0,0,3,0,0],[0,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,0,0,1]]
#cov2 =[[10,0,0,0,0,0,0,0,0,1],[0,9,0,0,0,0,0,0,2,0],[0,0,8,0,0,0,0,3,0,0],[0,0,0,7,0,0,4,0,0,0],[0,0,0,0,6,5,0,0,0,0],[0,0,0,0,5,6,0,0,0,0],[0,0,0,4,0,0,7,0,0,0],[0,0,3,0,0,0,0,8,0,0],[0,2,0,0,0,0,0,0,9,0],[1,0,0,0,0,0,0,0,0,10]]
#cov3 =[[1,0,0,0,0,0,0,0,0,10],[0,2,0,0,0,0,0,0,9,0],[0,0,3,0,0,0,0,8,0,0],[0,0,0,4,0,0,7,0,0,0],[0,0,0,0,5,6,0,0,0,0],[0,0,0,0,6,5,0,0,0,0],[0,0,0,7,0,0,4,0,0,0],[0,0,8,0,0,0,0,3,0,0],[0,9,0,0,0,0,0,0,2,0],[10,0,0,0,0,0,0,0,0,1]]

G1A = np.random.multivariate_normal(mean1A,cov1,100).T
G2A = np.random.multivariate_normal(mean2A,cov2,420).T
G3A = np.random.multivariate_normal(mean3A,cov3,480).T
G1B = np.random.multivariate_normal(mean1B,cov1,100).T
G2B = np.random.multivariate_normal(mean2B,cov2,420).T
G3B = np.random.multivariate_normal(mean3B,cov3,480).T
"""
fig =plt.figure()
plt.scatter(G1A,marker ='+')
plt.scatter(G1B,c='green',marker ='o')
plt.show()
fig =plt.figure()
plt.scatter(G2A[:,0],G2A[:,1],marker ='+')
plt.scatter(G2B[:,0],G2B[:,1],c='green',marker ='o')
plt.show()
fig =plt.figure()
plt.scatter(G3A[:,0],G3A[:,1],marker ='+')
plt.scatter(G3B[:,0],G3B[:,1],c='green',marker ='o')
plt.show()
"""
y=[]
X=[]
i=0
for i in range(0,100):
  y.append([0])
  X.append([G1A[0][i],G1A[1][i],G1A[2][i],G1A[3][i],G1A[4][i],G1A[5][i],G1A[6][i],G1A[7][i],G1A[8][i],G1A[9][i],0])
for i in range(0,420):
  y.append([0])
  X.append([G2A[0][i],G2A[1][i],G2A[2][i],G2A[3][i],G2A[4][i],G2A[5][i],G2A[6][i],G2A[7][i],G2A[8][i],G2A[9][i],0])
for i in range(0,480):
  y.append([0])
  X.append([G3A[0][i],G3A[1][i],G3A[2][i],G3A[3][i],G3A[4][i],G3A[5][i],G3A[6][i],G3A[7][i],G3A[8][i],G3A[9][i],0])
for i in range(0,100):
  y.append([1])
  X.append([G1B[0][i],G1B[1][i],G1B[2][i],G1B[3][i],G1B[4][i],G1B[5][i],G1B[6][i],G1B[7][i],G1B[8][i],G1B[9][i],1])
for i in range(0,420):
  y.append([1])
  X.append([G2B[0][i],G2B[1][i],G2B[2][i],G2B[3][i],G2B[4][i],G2B[5][i],G2B[6][i],G2B[7][i],G2B[8][i],G2B[9][i],1])
for i in range(0,480):
  y.append([1])
  X.append([G3B[0][i],G3B[1][i],G3B[2][i],G3B[3][i],G3B[4][i],G3B[5][i],G3B[6][i],G3B[7][i],G3B[8][i],G3B[9][i],1])

#meanB=[2,0,0,0,0,0,0,0,0,0]
#X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = np.random.multivariate_normal(meanB,cov,1000).T
#classB = np.random.multivariate_normal(meanB,cov,1000)
#print classB
#  X.append([X1[i],X2[i],X3[i],X4[i],X5[i],X6[i],X7[i],X8[i],X9[i],X10[i]])
X=np.matrix(X)
y=np.matrix(y)
X_train, X_test = train_test_split(X,test_size=0.4, random_state=0)
np.savetxt("Data/DS2-train.csv", X_train, delimiter=",")
np.savetxt("Data/DS2-test.csv", X_test, delimiter=",")
#np.savetxt("DS2/train_label.csv", Y_train, delimiter=",")
#np.savetxt("DS2/test_label.csv", Y_test, delimiter=",")
