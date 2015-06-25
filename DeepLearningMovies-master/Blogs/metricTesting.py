import csv
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import scipy
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


######


import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
imp = Imputer(missing_values="NaN")


#vectorArray = np.load("vectorArray.npy")
labelsArray = np.load("labelsArray.npy")
meanVector = np.load("meanVector.npy")
gMeanVector = np.load("gMeanVector.npy")
medianVector = np.load("medianVector.npy")
minVector = np.load("minVector.npy")
maxVector = np.load("maxVector.npy")
stdVector = np.load("stdVector.npy")

#meanVector = np.reshape(meanVector,(2845,300))
#gMeanVector = np.reshape(gMeanVector,(2845,300))
#medianVector = np.reshape(medianVector,(2845,300))
#minVector = np.reshape(minVector,(2845,300))
#maxVector = np.reshape(maxVector,(2845,300))
#stdVector = np.reshape(stdVector,(2845,300))

vectList = [meanVector,medianVector,minVector,maxVector,stdVector]
vectListNames = ['meanVector','medianVector','minVector','maxVector','stdVector']
label = labelsArray

n = 0
for llist in vectList:
    name = vectListNames[n]

    print np.count_nonzero(~np.isnan(llist))
    print llist.size

    llist = np.ma.masked_invalid(llist)

    print "list has ", llist.count(), "unmasked valuse"

    print "new ", np.count_nonzero(~np.isnan(llist))
    print llist.size

    #print llist

    forest = RandomForestClassifier( n_estimators = 1000 )
    train_data = llist[0:2200,:]
    train_labels = label[0:2200]
    #new_train_data = imp.fit_transform(train_data)
    new_train_data = train_data
    #print new_train_data
    #new_train_labels = imp.fit_transform(train_labels)
    #print "trainlable size ", new_train_labels.shape

    print "Fitting a random forest to labeled training data...", name
    try:
        forest = forest.fit(new_train_data,train_labels)

    # Test & extract results
        result = forest.predict( llist[2201:,:])

        score = forest.score(llist[2201:,:],label[2201:])


    # Write the test results
        print "forest score ",name," ", score
    except ValueError:
        print "fucking values"

    clf = svm.SVC()
    clf.fit(new_train_data,train_labels)


    score2 = clf.score(llist[2201:,:],label[2201:])

    print "svm score ",name, " ", score2

    gnb = GaussianNB()

    y_preditc = gnb.fit(new_train_data,train_labels)
    score3 = gnb.score(llist[2201:,:],label[2201:])

    print "baesian classifier ",name," ", score3

    n+=1
