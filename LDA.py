from scipy import linalg
import numpy as np
import pylab as pl

from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

training_data = np.loadtxt(open('data/contest-only-train.csv',"rb"),delimiter=",")

y = training_data[:,training_data.shape[1]-1]
X = training_data[:,0:training_data.shape[1]-1]

X = normalize(X, norm='l2', axis=1)


# LDA
lda = LDA()
y_pred = lda.fit(X, y, store_covariance=True).predict(X)

print 'LDA'
target_names = ['class1', 'class2']
print(classification_report(y, y_pred, target_names=target_names))

"""
# LDA
qda = QDA(reg_param = 0.5)
y_pred = qda.fit(X, y, store_covariances=True).predict(X)

print 'QDA'
target_names = ['class1', 'class2']
print(classification_report(y, y_pred, target_names=target_names))
print 'RDA'

#RDA
for i in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
	rda = QDA(reg_param = i)
	y_pred = rda.fit(X, y, store_covariances=True).predict(X)
	print i
	target_names = ['class1', 'class2']
	print(classification_report(y, y_pred, target_names=target_names))



# QDA
qda = QDA()
y_pred = qda.fit(X, y, store_covariances=True).predict(X)

print 'QDA'
target_names = ['class1', 'class2']
print(classification_report(y, y_pred, target_names=target_names))


"""
