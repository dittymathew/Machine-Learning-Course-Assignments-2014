import numpy as np
from sklearn.cross_validation import train_test_split
from StringIO import StringIO
import pandas as pd

f = open('datanew/contest_train.csv').read()
data = np.genfromtxt(StringIO(f), delimiter=",")
df = pd.DataFrame(data)

#filling with mean
new_data = np.matrix(df.fillna(df.median()))

#filling with interpolate
#s= pd.Series(df)
#print s
#new_data = np.matrix(df.fillna(s.interpolate('linear')))
#print new_data.shape


# complete train
np.savetxt('datanew/contest_only_train_median.csv', new_data, delimiter=',')

#for splitting
"""
X_A_Train1,  X_A_Test1 = train_test_split(new_data, test_size=0.10, random_state=42)

print X_A_Train1.shape
print X_A_Test1.shape

np.savetxt('data/contest-train1_ip.csv', X_A_Train1, delimiter=',')
np.savetxt('data/contest-test1_ip.csv', X_A_Test1, delimiter=',')
"""

#preprocessing
#missing values
#oversampling and under sampling
