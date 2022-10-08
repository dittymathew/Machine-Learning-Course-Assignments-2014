import numpy as np
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import operator

training_data = np.loadtxt(open('data/contest-only-train_ip_os.csv',"rb"),delimiter=",")
testing_data = np.loadtxt(open('data/contest-only-test_ip.csv',"rb"),delimiter=",")

y = training_data[:,training_data.shape[1]-1]
X = training_data[:,0:training_data.shape[1]-1]
model = ExtraTreesClassifier()
model.fit(X,y)
m_fimp =model.feature_importances_

f_imp={}
for i in range(0,len(m_fimp)):
  f_imp[i] = m_fimp[i]
sort_f=sorted(f_imp.iteritems(),key=operator.itemgetter(1)) # sorted in increasing order of importance
#print sort_f
sel_f =[f[0] for f in sort_f]
i=len(sel_f)-1
n=0
selectFeatures =[]
while i>=0:
  if n== 500:
    break
 # print sel_f[i],
  selectFeatures.append(sel_f[i])
  n +=1
  i -=1
selectFeatures.append(1897)
new_testdata = np.empty((1500,0))
new_traindata = np.empty((4200,0))
test_labels =np.zeros((1500,1))
for f in selectFeatures:
#    print new_testdata.shape,np.matrix(testing_data[:,f]).shape
    if f ==1897:
      new_testdata =np.column_stack((new_testdata,test_labels))
    else:
      new_testdata =np.column_stack((new_testdata,np.matrix(testing_data[:,f]).T))
#    print new_traindata.shape,np.matrix(training_data[:,f]).shape
    new_traindata =np.column_stack((new_traindata,np.matrix(training_data[:,f]).T))
np.savetxt('data/extratreeSelect/unlabelled-given-interpolate.csv', new_testdata, delimiter=',')
np.savetxt('data/extratreeSelect/labelled-given-interpolate-os-norm.csv', new_traindata, delimiter=',')
#print selectFeatures
