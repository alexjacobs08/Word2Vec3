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


#make patient vectors
#take each note for a single patient ID and merge them into a single vector for each patient
#merge them using different columneise statistical metrics, mean, median, ect...
#flatten the matrix to make a single 1XN matrix for each patient
# with N being a fucking number jesus i dont know

#kaggle vector funcitons

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000. == 0.:
          # print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

def unfusy_reader(csv_reader):
    while True:
        try:
            yield next(csv_reader)
        except csv.Error:
            print "fuck"
            continue

#import model
fname = "300features_40minwords_10context"
model = Word2Vec.load(fname)



#import data
with open("../../../../blogData/output.tsv",'rb') as f:

    reader = unfusy_reader(csv.reader(f, delimiter='\t'))
    x = 0
    count = 0
    newList = []
    justVectors = []
    justGender = []
    for row in reader:
        if count % 1000 == 0:
            print "row ", count

        idd = row[0]
        gender = row[1]
        blog = row[2]

        if x ==0:
            lastId = idd
            vectList = []
            x = 1

        if idd == lastId:
            vectList.append(blog)
            lastId = idd
        else:
            vectList = getCleanReviews(vectList)

            blogArrays = getAvgFeatureVecs(vectList, model, 300)

            A = np.array(blogArrays) #create numpy array from vectList
            m,n = A.shape

            patientVector = np.zeros((6,n),dtype="float32")#prealllocate for speed
            #8 is for number of metrics
            #n is for number of features, n should equal num_features

            for col in xrange(n):
                patientVector[0,col] = np.nanmean(A[:,col])
                patientVector[1,col] = scipy.stats.mstats.gmean(A[:,col])
                patientVector[2,col] = np.nanmedian(A[:,col])
                patientVector[3,col] = np.nanmin(A[:,col])
                patientVector[4,col] = np.nanmax(A[:,col])
                patientVector[5,col] = np.nanstd(A[:,col])

            #flatten patient vector

            flattenedPatientVector = np.ravel(patientVector)


            newRow = [idd,gender,flattenedPatientVector]
            justVectors.append(flattenedPatientVector)
            justGender.append(gender)
            newList.append(newRow)

            vectList = []
            vectList.append(blog)
            lastId = idd




        x = 1
        count += 1
    f.close()

    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(newList)
    f.close()
    with open("jutsVectors.csv","wb") as f:
        writer = csv.writer(f)
        writer.writerows(justVectors)
    f.close()
    with open("justGender.csv","wb") as f:
        writer = csv.writer(f)
        writer.writerows(justGender)
    f.close()



    newHead = pd.DataFrame(newList)
    print newHead.head(10)
