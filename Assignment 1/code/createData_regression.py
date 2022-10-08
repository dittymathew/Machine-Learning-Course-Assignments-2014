# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 11:36:40 2014

@author: ditty
"""
import numpy as np
from pandas import DataFrame
from StringIO import StringIO
from sklearn.cross_validation import train_test_split
import pandas as pd

data = open("communities_processed.csv").read()
d =DataFrame(np.genfromtxt(StringIO(data), delimiter=","))
s= pd.Series(d)
filled_data = np.array(d.fillna(s.interpolate('linear')))
#filled_data = np.array(d.fillna(d.mean()))
#X = filled_data[:,0:126]
#y = filled_data[:,126]

d1_train,d1_test = train_test_split(filled_data,test_size=0.2, random_state=0)
d2_train,d2_test = train_test_split(filled_data,test_size=0.2, random_state=500)
d3_train,d3_test = train_test_split(filled_data,test_size=0.2, random_state=1000)
d4_train,d4_test = train_test_split(filled_data,test_size=0.2, random_state=1500)
d5_train,d5_test = train_test_split(filled_data,test_size=0.2, random_state=1900)
np.savetxt("Data/CandC-train1.csv", d1_train, delimiter=",")
np.savetxt("Data/CandC-test1.csv", d1_test, delimiter=",")
np.savetxt("Data/CandC-train2.csv", d2_train, delimiter=",")
np.savetxt("Data/CandC-test2.csv", d2_test, delimiter=",")
np.savetxt("Data/CandC-train3.csv", d3_train, delimiter=",")
np.savetxt("Data/CandC-test3.csv", d3_test, delimiter=",")
np.savetxt("Data/CandC-train4.csv", d4_train, delimiter=",")
np.savetxt("Data/CandC-test4.csv", d4_test, delimiter=",")
np.savetxt("Data/CandC-train5.csv", d5_train, delimiter=",")
np.savetxt("Data/CandC-test5.csv", d5_test, delimiter=",")
#print D1[0].shape
#dd= (D1[0],D1[2])
#D1_train =np.concatenate((D1[0],D1[2]),axis=1)
#data1 =data.filled(data.mean())
