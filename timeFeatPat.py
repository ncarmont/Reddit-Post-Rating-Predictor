#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
data= list()
with open('redditSubmissions.csv', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


# In[8]:


import numpy as np
import scipy.optimize
import random
from collections import defaultdict
import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

import time
import re
from collections import defaultdict
ratingPerHour = defaultdict(int)
timeCounter = defaultdict(int)

def returnRating(upvotes, totVotes):
    if totVotes is None: return 0
    if(int(totVotes)==0):
        return 0
    return int(upvotes)/int(totVotes)

def convertToHour(dateT):
    ISOtime = re.sub('......$', '', dateT)
    while(len(ISOtime)>19):
        ISOtime = re.sub('.$', '', ISOtime)
    timeStruct= time.strptime(ISOtime, "%Y-%m-%dT%H:%M:%S")
    return timeStruct[3]

for d in data: 
    ratingMetric = returnRating(d['number_of_upvotes'],d['total_votes'])
    dateTime = d['rawtime']
    if(dateTime):
        hour = convertToHour(dateTime)
        ratingPerHour[hour]+= int(ratingMetric)
        timeCounter[hour]+=1
        
avgRating = list()
for (hour,tot) in ratingPerHour.items():
    avgScore.append((hour,tot/timeCounter[hour]))
    
avgRating.sort(key=lambda tup: tup[1])
avgRating.reverse()
avgRatingDict = dict(avgRating)
             
def avgHourRating(ISOtime):
    feat =[1]
    if ISOtime is None: 
        feat.append(0)
    else:
        hour =convertToHour(ISOtime)
        totRatings =dict(ratingPerHour)
        totRating = int(totRatings[hour])
        feat.append(totRating)
    return feat

X = [avgHourRating(d['rawtime']) for d in data]
y = [returnRating(d['number_of_upvotes'],d['total_votes']) for d in data]

halfLenData = int(len(data)/2)
X_train = X[:halfLenData]
y_train = y[:halfLenData]

X_test = X[halfLenData:]
y_test = y[halfLenData:]

theta1,residuals,rank,s = np.linalg.lstsq(X_train, y_train,rcond= -1)
predictionsTrn = [sum(x*theta1) for x in X_train]
predictionsTst = [sum(x*theta1) for x in X_test]

MSELinRegTrn = np.square(np.subtract(y_train,predictionsTrn)).mean() 
MSELinRegTst = np.square(np.subtract(y_test,predictionsTst)).mean() 
print("Lin reg for max total score: ")
print("MSE for Training " +str(MSELinRegTrn))
print("MSE for Test "+str(MSELinRegTst))


# In[9]:


def printFirst10(arr):
    counter = 0
    for a in arr:
        print(a)
        counter+=1
        if(counter>10):
            break
printFirst10(predictionsTrn)
printFirst10(predictionsTst)

