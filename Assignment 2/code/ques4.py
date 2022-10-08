from sklearn import datasets
from sklearn.lda import LDA
import numpy as np
from sklearn.qda import QDA
#from sklearn.rda import RDA
import pylab as pl
import matplotlib as mpl
from scipy import linalg


iris = datasets.load_iris()
X=iris.data[:,2:4]
y=iris.target
target_names = iris.target_names
print target_names

lda = LDA(n_components=1)
y_pred =lda.fit(X,y,store_covariance=True).predict(X)

qda = QDA()
y_pred = qda.fit(X, y, store_covariances=True).predict(X)


rda = QDA(reg_param=0.1)
y_pred = rda.fit(X, y, store_covariances=True).predict(X)

xx, yy = np.meshgrid(np.linspace(0, 10, 500), np.linspace(0, 4.5, 500))
X_grid = np.c_[xx.ravel(), yy.ravel()]
zz_lda = lda.predict_proba(X_grid)[:,1].reshape(xx.shape)
zz_qda = qda.predict_proba(X_grid)[:,1].reshape(xx.shape)
zz_rda = rda.predict_proba(X_grid)[:,1].reshape(xx.shape)

pl.figure()
pl.contourf(xx, yy, zz_lda > 0.5, alpha=0.5)
pl.scatter(X[y==0,0], X[y==0,1], c='b', label=target_names[0])
pl.scatter(X[y==1,0], X[y==1,1], c='r', label=target_names[1])
pl.scatter(X[y==2,0], X[y==2,1], c='g', label=target_names[2])
pl.contour(xx, yy, zz_lda,[0.5],  linewidths=2., colors='k')
pl.axis('tight')
pl.title('Linear Discriminant Analysis')
pl.show()

pl.contourf(xx, yy, zz_qda > 0.5, alpha=0.5)
pl.scatter(X[y == 0, 0], X[y == 0, 1], c='b', label=target_names[0])
pl.scatter(X[y == 1, 0], X[y == 1, 1], c='g', label=target_names[1])
pl.scatter(X[y == 2, 0], X[y == 2, 1], c='r', label=target_names[1])
pl.contour(xx, yy, zz_qda, [0.5], linewidths=2., colors='k')
pl.axis('tight')
pl.title('Quadratic Discriminant Analysis')
pl.show()

pl.contourf(xx, yy, zz_rda > 0.5, alpha=0.5)
pl.scatter(X[y == 0, 0], X[y == 0, 1], c='b', label=target_names[0])
pl.scatter(X[y == 1, 0], X[y == 1, 1], c='g', label=target_names[1])
pl.scatter(X[y == 2, 0], X[y == 2, 1], c='r', label=target_names[1])
pl.contour(xx, yy, zz_rda, [0.5], linewidths=2., colors='k')
pl.axis('tight')
pl.title('Regularized Discriminant Analysis lamda=0.1')
pl.show()
