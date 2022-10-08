import numpy as np
from sklearn.cross_validation import train_test_split
from StringIO import StringIO
from sklearn.preprocessing import normalize
import pandas as pd

f = open('data/contest_only_test.csv').read()
data = np.genfromtxt(StringIO(f), delimiter=",")
df = pd.DataFrame(data)

#filling with mean
#new_data = np.matrix(df.fillna(df.mean()))

#filling with interpolate
s= pd.Series(df)
#print s
new_data = np.matrix(df.fillna(s.interpolate('linear')))
print new_data.shape
X_norm = normalize(new_data, norm='l2', axis=1)


# complete train
np.savetxt('data/contest-only-test_ip.csv', X_norm, delimiter=',')



#preprocessing
#missing values
#oversampling and under sampling
