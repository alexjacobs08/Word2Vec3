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
import re
from bs4 import BeautifulSoup
from KaggleWord2VecUtility import KaggleWord2VecUtility


######


import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
imp = Imputer(missing_values="NaN")



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

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


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
    meanVectors = []
    gMeanVectors = []
    medianVectors = []
    minVectors = []
    maxVectors = []
    stdVectors =[]

    n = 300

    #patientVector = np.zeros((6,n),dtype="float64")#prealllocate for speed
    meanVector = np.zeros((1,n),dtype="float64")
    gMeanVector = np.zeros((1,n),dtype="float64")
    medianVector = np.zeros((1,n),dtype="float64")
    minVector = np.zeros((1,n),dtype="float64")
    maxVector = np.zeros((1,n),dtype="float64")
    stdVector = np.zeros((1,n),dtype="float64")

    for row in reader:
        if count % 1000 == 0:
            print "row ", count

        idd = row[0]
        gender = row[1]
        blog = row[2]
        #print blog

        if x == 0:
            lastId = idd
            vectList = []
            clean_reviews = []
            x = 1

        if idd == lastId:
            vectList.append(blog)
            lastId = idd
            genderOld = gender
        else:
            #vectList = getCleanReviews(vectList)

            for i in xrange(len(vectList)):
                clean_reviews.append((review_to_words(vectList[i])))
            #print vectList

            A = getAvgFeatureVecs(clean_reviews, model, 300)
            #print blogArrays.shape

            #print A[i,:]


            justGender.append(genderOld)

            #print scipy.stats.mstats.gmean(A[:,:]).reshape(1,300)
            gMean = np.array(scipy.stats.mstats.gmean(A, axis=0))
            gMeanVector = np.vstack((gMeanVector, gMean))

            median = np.array(np.nanmedian(A, axis=0))
            medianVector = np.vstack((medianVector, median))

            mean = np.array(np.mean(A,axis=0))
            meanVector = np.vstack((meanVector,mean))

            min = np.array(np.nanmin(A,axis=0))
            minVector = np.vstack((minVector, min))

            max = np.array(np.nanmax(A,axis=0))
            maxVector = np.vstack((maxVector,max))

            std = np.array(np.nanstd(A,axis=0))
            stdVector = np.vstack((stdVector,std))






            """



            #8 is for number of metrics
            #n is for number of features, n should equal num_features


            for col in xrange(n):
                patientVector[0,col] = np.nanmean(A[0:m,col], dtype="float64")
                #meanVector[0,col] = np.nanmean(A[:,col],dtype="float64")

                patientVector[1,col] = np.float64(scipy.stats.mstats.gmean(A[:,col]))
                gMeanVector[0,col] = np.float64(scipy.stats.mstats.gmean(A[:,col]))

                patientVector[2,col] = np.float64(np.nanmedian(A[:,col]))
                medianVector[0,col] = np.float64(np.nanmedian(A[:,col]))

                patientVector[3,col] = np.float64(np.nanmin(A[:,col]))
                minVector[0,col] = np.float64(np.nanmin(A[:,col]))

                patientVector[4,col] = np.float64(np.nanmax(A[:,col]))
                maxVector[0,col] = np.float64(np.nanmax(A[:,col]))

                patientVector[5,col] = np.float64(np.nanstd(A[:,col]))
                stdVector[0,col] = np.float64(np.nanstd(A[:,col]))

            #flatten patient vecto



            flattenedPatientVector = np.ravel(patientVector)


            newRow = [idd,gender,flattenedPatientVector]

            justVectors.append(flattenedPatientVector)
            justGender.append(gender)
            meanVectors.append(patientVector[0,:])
            gMeanVectors.append(patientVector[1,:])
            medianVectors.append(patientVector[2,:])
            minVectors.append(patientVector[3,:])
            maxVectors.append(patientVector[4,:])
            stdVectors.append(patientVector[5,:])
            """


            vectList = []
            vectList.append(blog)
            lastId = idd




        x = 1
        count += 1
    f.close()

    labelArray = np.asarray(justGender)


    np.save("labelsArray",labelArray)
    np.save("meanVector",meanVector)
    np.save("gMeanVector",gMeanVector)
    np.save("medianVector",medianVector)
    np.save("minVector",minVector)
    np.save("maxVector",maxVector)
    np.save("stdVector",stdVector)


    print "just gender length", len(justGender)

    print "vectors shape", meanVectors.shape



    """

    vectList = [justGender,justVectors,meanVectors,gMeanVectors,medianVectors,minVectors,maxVectors,stdVectors]

    for vect in vectList:
        del vect[775:777]



    vectorsArray = np.asarray(justVectors)
    labelArray = np.asarray(justGender)
    meanArray = np.asarray(meanVectors)
    gMeanArray = np.asarray(gMeanVectors)
    medianArray = np.asarray(medianVectors)
    minArray = np.asarray(minVectors)
    maxArray = np.asarray(maxVectors)
    stdArray = np.asarray(stdVectors)



    np.save("vectorArray",vectorsArray)
    np.save("labelsArray",labelArray)
    np.save("meanVector",meanArray)
    np.save("gMeanVector",gMeanArray)
    np.save("medianVector",medianArray)
    np.save("minVector",minArray)
    np.save("maxVector",maxArray)
    np.save("stdVector",stdArray)


labelsArray = np.load("labelsArray.npy")
meanVector = np.load("meanVector.npy")
gMeanVector = np.load("gMeanVector.npy")
medianVector = np.load("medianVector.npy")
minVector = np.load("minVector.npy")
maxVector = np.load("maxVector.npy")
stdVector = np.load("stdVector.npy")

meanVector = np.reshape(meanVector,(2845,300))
gMeanVector = np.reshape(gMeanVector,(2845,300))
medianVector = np.reshape(medianVector,(2845,300))
minVector = np.reshape(minVector,(2845,300))
maxVector = np.reshape(maxVector,(2845,300))
stdVector = np.reshape(stdVector,(2845,300))



"""

vectList = [meanVector,gMeanVector,medianVector,minVector,maxVector,stdVector]
vectListNames = ['meanVector','gMeanVector','medianVector','minVector','maxVector','stdVector']
label = labelArray

n = 0
for llist in vectList:
    name = vectListNames[n]

    print np.count_nonzero(~np.isnan(llist))
    print llist.size

    where_are_NaNs = np.isnan(llist)
    llist[where_are_NaNs] = 0

    #print llist

    forest = RandomForestClassifier( n_estimators = 1000 )
    train_data = llist[0:2200,:]
    train_labels = label[0:2200]
    new_train_data = imp.fit_transform(train_data)
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



    """
    try:
        a = np.array( meanVectors )
        b = np.array( gMeanVectors )
        c = np.array( medianVectors )
        d = np.ndarray( minVectors )
        e = np.ndarray( maxVectors )
        f = np.ndarray( stdVectors )

        np.save("meanVector",a)
        np.save("gMeanVector",b)
        np.save("medianVector",c)
        np.save("minVector",d)
        np.save("maxVector",e)
        np.save("stdVector",f)

        np.savetxt("meanVector.csv",a, delimiter=',')
        np.savetxt("gMeanVector.csv",b, delimiter=',')
        np.savetxt("medianVector.csv",c, delimiter=',')
        np.savetxt("minVector.csv",d, delimiter=',')
        np.savetxt("maxVector.csv",e, delimiter=',')
        np.savetxt("stdVector.csv",f, delimiter=',')


    except TypeError:
        print"fucking shit a type error"

    try:
        a = np.ndarray( meanVectors )
        b = np.ndarray( gMeanVectors )
        c = np.ndarray( medianVectors )
        d = np.ndarray( minVectors )
        e = np.ndarray( maxVectors )
        f = np.ndarray( stdVectors )

        np.savetxt("meanVector1.csv",a, delimiter=',')
        np.savetxt("gMeanVector1.csv",b, delimiter=',')
        np.savetxt("medianVector1.csv",c, delimiter=',')
        np.savetxt("minVector1.csv",d, delimiter=',')
        np.savetxt("maxVector1.csv",e, delimiter=',')
        np.savetxt("stdVector1.csv",f, delimiter=',')

        np.save("meanVector1",a)
        np.save("gMeanVector1",b)
        np.save("medianVector1",c)
        np.save("minVector1",d)
        np.save("maxVector1",e)
        np.save("stdVector1",f)
    except TypeError:
        print"fucking shit another type error"


    try:

        with open("output3.csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerows(newList)
        f.close()

        with open("jutsVectors3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(justVectors)
        f.close()
        with open("justGender3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(justGender)
        f.close()
        with open("meanVector3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(meanVectors)
        f.close()
        with open("gMeanVector3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(gMeanVectors)
        f.close()
        with open("medianVector3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(medianVectors)
        f.close()
        with open("minVector3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(minVectors)
        f.close()
        with open("stdVector3.csv","wb") as f:
            writer = csv.writer(f)
            writer.writerows(stdVectors)
        f.close()

    except TypeError:
        print "lsdkfjsdlkfjalsfkjsadfsa"








    newHead = pd.DataFrame(newList)
    print newHead.head(10)


    """