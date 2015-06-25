import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
imp = Imputer(missing_values="NaN")

from KaggleWord2VecUtility import KaggleWord2VecUtility

vectors = np.loadtxt('jutsVectors.csv',delimiter=',',dtype='float64')
label = np.loadtxt('justGender.csv',delimiter=',',dtype='float64')
meanVectors = np.loadtxt('meanVector.csv',delimiter=',',dtype='float64')
gMeanVectors = np.loadtxt('gMeanVector.csv',delimiter=',',dtype='float64')
medianVectors = np.loadtxt('medianVector.csv',delimiter=',',dtype='float64')
minVectors = np.loadtxt('minVector.csv',delimiter=',',dtype='float64')
#maxVectors = np.loadtxt('maxVector.csv',delimiter=',',dtype='float64')
stdVectors = np.loadtxt('stdVector.csv',delimiter=',',dtype='float64')

vectList = [vectors,meanVectors,gMeanVectors,medianVectors,minVectors,stdVectors]




print "vector shape ", vectors.shape
print "label shape ", label.shape
print "meanVector shape ", meanVectors.shape
#ind = np.where(~np.isnan(vectors))[0]
#first, last = ind[0], ind[-1]
#vectors[:first] = vectors[first]
#vectors[last + 1:] = vectors[last]

#where_are_NaNs = np.isnan(vectors)
#vectors[where_are_NaNs] = 0
#print "where are nans ", where_are_NaNs.shape

#print "vector shape new ", vectors.shape

# ****** Fit a random forest to the training set, then make predictions
#
# Fit a random forest to the training data, using 100 trees

for llist in vectList:

    forest = RandomForestClassifier( n_estimators = 1000 )
    train_data = llist[0:2200,:]
    train_labels = label[0:2200]
    new_train_data = imp.fit_transform(train_data)
    print new_train_data
    #new_train_labels = imp.fit_transform(train_labels)
    #print "trainlable size ", new_train_labels.shape

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(new_train_data,train_labels)

    # Test & extract results
    result = forest.predict( llist[2201:,:])

    score = forest.score(llist[2201:,:],label[2201:])

    # Write the test results
    print "forest score ",llist," ", score

    clf = svm.SVC()
    clf.fit(new_train_data,train_labels)


    score2 = clf.score(llist[2201:,:],label[2201:])

    print "svm score ",llist, " ", score2

    gnb = GaussianNB()

    y_preditc = gnb.fit(new_train_data,train_labels)
    score3 = gnb.score(llist[2201:,:],label[2201:])

    print "baesian classifier ",llist," ", score3