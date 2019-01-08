#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
data= list()
with open('redditSubmissions.csv', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


# In[2]:


# Cosine similarity!
# user-user recc for rating with cold case baselines and return avg for 1 user-resub
from collections import defaultdict
import math
import numpy as np
from collections import Counter
dataWimg = list()
for d in data:
    if (d['#image_id']):
        if (d['number_of_upvotes']):
            if (d['total_votes']):
                dataWimg.append(d)
                
def returnRating(upvotes, totVotes):
    if(int(totVotes)==0):
        return 0
    return int(upvotes)/int(totVotes)
            
X = [(d['username'],d['#image_id']) for d in dataWimg]
y = [returnRating(d['number_of_upvotes'],d['total_votes']) for d in dataWimg]


halfLenData = len(dataWimg)/2
# print(halfLenData)

trnData = dataWimg [:66153]
X_train = X[:66153]
y_train = y[:66153]

X_test = X[66153:]
y_test = y[66153:]

usersImageSubs = defaultdict(list)
imageUserSubs = defaultdict(list)

for d in trnData:
    user = d['username']
    img = d['#image_id']
    usersImageSubs[user].append(img)
    imageUserSubs[img].append(user)
    
userUserDict = defaultdict(list) 
userSim = defaultdict(dict)    
for user in usersImageSubs:
    for img in usersImageSubs[user]:
        userUserDict[user]+= [u for u in imageUserSubs[img] if ((u !=user) & (u!=''))]
    userSim[user] = Counter(userUserDict[user])
        
def printFirst10(arr):
    counter = 0
    for a in arr:
        print(a)
        counter+=1
        if(counter>10):
            break
    

# printFirst10(userJaccard.items())

ratings = defaultdict(dict)
userCount = defaultdict(int)
userTotRating = defaultdict(int)
imageCount = defaultdict(int)
imageTotRating = defaultdict(int)
totRating =0
users = set()
images = set()
for d in trnData:
    ratings[d['username']][d['#image_id']] = returnRating(d['number_of_upvotes'],d['total_votes'])
    users.add(d['username'])
    images.add(d['#image_id'])
    imageCount[d['#image_id']] +=1 
    imageTotRating[d['#image_id']] += returnRating(d['number_of_upvotes'],d['total_votes'])
    userCount[d['username']] +=1 
    userTotRating[d['username']] += returnRating(d['number_of_upvotes'],d['total_votes'])
    totRating+=returnRating(d['number_of_upvotes'],d['total_votes'])
    
avgRating = totRating/len(trnData)
    
counterrr= 0
    
userCosine = defaultdict(dict)
for user in userSim:
    if(user):
        for u2 in userSim[user]:
            dotProd = 0
            if(u2):
                for i in usersImageSubs[user]:
                    if (i in usersImageSubs[u2]):
                        dotProd+= ratings[user][i]*ratings[u2][i]# should be values of that image sim mult together
                        counterrr+=1
                userCosine[user][u2] = dotProd/(math.sqrt(len(usersImageSubs[user]))*math.sqrt(len(usersImageSubs[u2])))
                
                
    # if user not seen before, return img average 
# Calculate img averages

avgImageRating = defaultdict(int)
for i in imageTotRating:
    avgImageRating[i] = imageTotRating[i]/imageCount[i]
    
avgUserRating = defaultdict(int)
for i in userTotRating:
    avgUserRating[i] = userTotRating[i]/userCount[i]
    

# if img not seen before, return user average

# if none seen before, return overall average (for users with 1 post)

def weightedUserSimScore(user, img):
    rating = 0
    if((user in users) & (img in images)):
        weightedSum = 0
        sumSim = 0
        for (u,v) in userCosine[user].items():
            if(img in usersImageSubs[u]):
                weightedSum+= ratings[u][img]*v
                sumSim+=v
        if(sumSim==0):
            sumSim = 1
        return (weightedSum/sumSim)
    elif (img in images):
        return avgImageRating[img]
    elif (user in users):
        return avgUserRating[user]
    else:
        return avgRating

predictionsTrn = [weightedUserSimScore(user,img) for (user,img) in X_train]

predictionsTst = [weightedUserSimScore(user,img) for (user,img) in X_test]


# In[3]:


MSELinRegTrn = np.square(np.subtract(y_train,predictionsTrn)).mean() 
MSELinRegTst = np.square(np.subtract(y_test,predictionsTst)).mean() 
print("User similarity Ratings (with cold case avg baseline): ")
print("MSE for Training " +str(MSELinRegTrn))
print("MSE for Test "+str(MSELinRegTst))

