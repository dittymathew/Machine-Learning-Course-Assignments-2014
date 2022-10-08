from PIL import Image
import os
import numpy as np

def get32bin(h):
  j=0
  f =[]
  while j<256:
    a =0
    for k in range(j,(j+8) ):
      a += h[k]
    a = float(a)/float(8)  # Mean need to verify
    f.append(a)
    j +=8
  return f

def getExtractFeature(d,data,output,n,label):
  for f in os.listdir(d):
    data.append([])
    img =Image.open(d+'/'+f)
    hist =img.histogram()
    r =hist[0:256]
    g =hist[256:512]
    b =hist[512:768]
    for f in get32bin(r):
      data[n].append(f)
    for f in get32bin(g):
      data[n].append(f)
    for f in get32bin(b):
      data[n].append(f)
    output.append(label)
    n +=1
  return (data,output,n)


category ={'coast':1,'forest':2,'insidecity':3,'mountain':4}
d1 ="data/"
X_train =[]
X_test =[]
nTrain =0
nTest =0
y_train=[]
y_test=[]
for d2 in os.listdir(d1):
  d =d1+d2+'/Train'
  (X_train,y_train,nTrain) = getExtractFeature(d,X_train,y_train,nTrain,category[d2])
  d =d1+d2+'/Test'
  (X_test,y_test,nTest) = getExtractFeature(d,X_test,y_test,nTest,category[d2])
X_train =np.array(X_train)
X_test =np.array(X_test)
y_train =np.array(y_train)
y_test =np.array(y_test)
train = np.column_stack((X_train,y_train))
test = np.column_stack((X_test,y_test))
print train
np.savetxt("DS2-train.csv", train, delimiter=",")
np.savetxt("DS2-test.csv", test, delimiter=",")
