import random
import os
from nltk.corpus import stopwords
import nltk.data
#nltk.download()
import re
from collections import Counter
from bs4 import BeautifulSoup
import numpy as np
import scipy
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer, normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import cross_validation
imp = Imputer(missing_values="NaN")
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import heapq
import math
from sklearn.cluster import KMeans
import time
from tempfile import TemporaryFile

# for debugging purposes
#np.set_printoptions(threshold=np.nan)

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #9000
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
    #featureVec = np.array(scipy.stats.mstats.gmean(featureVec, axis=0))
    #featureVec = np.array(np.median(featureVec, axis=0))
    #featureVec = np.array(np.max(featureVec, axis=0))
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




def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    #words = tokenizer.tokenize(words.strip())
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences



#fname = "w2vModelsize300"
num_features = 300
#model = Word2Vec.load(fname)



# data
# ../../../../txt_sentoken/pos contains 1000 movie reviews, each in its own .txt file, each review is of positive sentiment (1)
# ../../../../txt_sentoken/neg contains 1000 movie reviews, each in its own .txt file, each review is of negative sentiment (0)
positiveDataFile = "../../../../txt_sentoken/pos"
negativeDataFile = "../../../../txt_sentoken/neg"

#lists to store reviews initially
positiveReviews = []
negativeReviews = []
for i in os.listdir(positiveDataFile):
    with open(os.path.join(positiveDataFile,i),"rb") as f:
        #add each review to list in its entirety
        positiveReviews.append(f.read())
    f.close()

for i in os.listdir(negativeDataFile):
    with open(os.path.join(negativeDataFile,i),"rb") as f:
        negativeReviews.append(f.read())
    f.close()

corpus = list(positiveReviews + negativeReviews)

bowCorpus = []
for i in xrange(len(corpus)):
    bowCorpus.append(review_to_words(corpus[i]))

wNeg = re.findall(r'\w+', str([negativeReviews[i] for i in xrange(len(negativeReviews))]))#all of the words
wPos = re.findall(r'\w+', str([positiveReviews[i] for i in xrange(len(positiveReviews))]))

WordCountsDict = Counter(wNeg + wPos)#dictionary with words as keys and count as values
avg = np.mean(WordCountsDict.values())#avg number of words
std = np.std(WordCountsDict.values())

dist2 = (avg - 2*std)

wordsPos = []
wordsNeg = []
posReviewToList = []
negReviewToList = []
for i in xrange(len(positiveReviews)):
    review = review_to_wordlist(positiveReviews[i], remove_stopwords= True)
    wordsPos += review
    posReviewToList.append(review)

for i in xrange(len(negativeReviews)):
    review = review_to_wordlist(negativeReviews[i],remove_stopwords= True)
    wordsNeg += review
    negReviewToList.append(review)

words = sorted(list(set().union(wordsPos,wordsNeg)))#all words in reviews, minus stop words
#print len(words)



review_1= []
for i in xrange(len(corpus)):
    review_1 += review_to_sentences(corpus[i], tokenizer,remove_stopwords=True)

random.shuffle(review_1) #shuffle the corpus

#print "builing Word2Vec model..."
model = Word2Vec(review_1, size=num_features, window=15, workers=1)

centroids = []
x = 0


for word in words:

    if x == 0:
        try:
            wordArray = np.asarray(model[word])
            x = 1
        except:
            x = 0
            print word

    else:

        try:
            wordArray = np.vstack((wordArray,[np.asarray(model[word])]))
        except:
            nsdfks = 0

nclusters = [100,200,300,400,500]

for clust in nclusters:
    nclusters = clust

    cluster = KMeans(init='k-means++', n_clusters=clust)

    cluster.fit_transform(wordArray)
    cluster_centers = cluster.cluster_centers_
    centroids.append((1,cluster_centers))

    dist = np.linalg.norm(cluster_centers[0] - cluster_centers[1])




    distanceArray = []

    #for center in cluster_centers:

    #for bin size, just find the closest centriod, it goes into that bin, increment that centroid by 1 and use the vector of centroids to classify
    clean_reviews_Pos = []
    clean_reviews_Pos_BOW = []
    clean_reviews_Neg = []
    clean_reviews_Neg_BOW = []
    clean_reviews_Neg_1 = []
    clean_reviews_Pos_1 = []
    print "cleaning and parsing data.."
    #print review_to_words(positiveReviews[1])
    review_size = 0
    for i in xrange(len(positiveReviews)):
        review_1 = review_to_wordlist(positiveReviews[i])  ###################################################################### change way words are used
        #clean_reviews_Pos_BOW.append(review_to_words(positiveReviews[i]))
        clean_reviews_Pos_1.append(review_1)

        review_ = review_to_wordlist(positiveReviews[i])
        clean_reviews_Pos.append(review_)

        review_size += len(review_)
        #print "review size", len(review_)

    for i in xrange(len(negativeReviews)):
        review_1 = review_to_wordlist(negativeReviews[i])  ###################################################################### change way words are used
        clean_reviews_Neg_BOW.append((review_to_words(negativeReviews[i])))
        clean_reviews_Neg_1.append(review_1)

        review_ = review_to_wordlist(negativeReviews[i])
        clean_reviews_Neg.append(review_)

        review_size += len(review_)

    totalsize = len(clean_reviews_Pos) + len(clean_reviews_Neg)


    test_Data = []

    for i in clean_reviews_Pos:
        test_Data.append((1,i))

    for i in clean_reviews_Neg:
        test_Data.append((0,i))

    print test_Data[0]

    allDataArrays = []

    print "creating all data array"
    startTime = time.time()

    print "len cluster data",len(test_Data)
    print "ncluster", nclusters

    for f in xrange(len(test_Data)):
        review = test_Data[f][1]
        label = test_Data[f][0]
        distanceArray = np.zeros(nclusters)
        #print len(review)
        for word in review:
            word = str(word)
            try:
                wordVec = model[word]
            except:
                t = 4


            distArr = []
            for clusterC in cluster_centers:
                distArr.append(np.linalg.norm(wordVec - clusterC))
            #print "distArr", distArr
            #print "distance min",np.argmin(distArr)
            distanceArray[np.argmin(distArr)] += 1
            #print distanceArray
        #print distanceArray
        allDataArrays.append((label,distanceArray))


    print "done creating all data arrays"
    stoptime = time.time()
    totaltime = stoptime - startTime
    print "time in seconds", totaltime

    #allArrayOutfile = TemporaryFile()


    #npAlldataArray = np.asarray(allDataArrays)
    #np.save(allArrayOutfile,npAlldataArray)

    print "outfile saved"



    posFeatures = getAvgFeatureVecs(clean_reviews_Pos_1, model, num_features)
    negFeatures = getAvgFeatureVecs(clean_reviews_Neg_1, model, num_features)




    #add label to review array
    posFeatures = np.insert(posFeatures,0,1,axis=1)#w2v
    negFeatures = np.insert(negFeatures,0,0,axis=1)#w2v

    #stack training vectors
    allData = np.vstack((posFeatures, negFeatures))#w2v
    labels = allData[:,0]
    allData = np.delete(allData, 0, axis=0)


    #shuffle data
    random.shuffle(test_Data)


    m = len(allDataArrays)
    print m
    k = 3
    crossValNum = 5
    folds = 4

    aucMatrix = np.zeros([k,folds])

    kf = cross_validation.KFold(m,n_folds = folds)
    fold = 0
    method = 2

    print "starting cross vald loop"

    startTime = time.time()

    for train_index, test_index in kf:
        fold +=1

        train = list(allDataArrays[f] for f in train_index)
        test = list(allDataArrays[f] for f in test_index)


        trainVectors = np.asarray(list(train[f][1] for f in xrange(len(train))))
        trainLables = np.asarray(list(train[f][0] for f in xrange(len(train))))

        testVectors = np.asarray(list(test[f][1] for f in xrange(len(test))))
        testLables = np.asarray(list(test[f][0] for f in xrange(len(test))))

        predicted = []
        true = []

        forest_size = 100
        forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
        #print trainVectors[0]
        forest = forest.fit(trainVectors,trainLables)
        oob_score =  forest.oob_score_
        print "oob score ", oob_score


        rcsList = [50,75,100,125,150,200]
        count = 0
        scoreKeeper = 0

        score = forest.score(testVectors, testLables)
        print "score ", score, " for fold ", fold

        """
        for i in xrange(len(testVectors)):


            distanceArray = np.zeros(nclusters)

            review = testVectors[i]
            Tuplabel = testLables[i]

            try:
                preditctedLabel = forest.predict(testVectors[i])
                count += 1
            except:
                print "fuck"
                g = False

            if preditctedLabel == Tuplabel:
                scoreKeeper+=1


            else:
                if Tuplabel == 1:
                    predicted.append(0)
                else:
                    predicted.append(1)

            true.append(Tuplabel)




        print "fold", fold
        print "scorekeeper",scoreKeeper
        print "count fold", count

        try:
            true = np.asarray(true)
            predicted = np.asarray(predicted)

            print "auc", roc_auc_score(true,predicted)
        except:
            print "auc didnt work"

        """

        stoptime = time.time()
        totaltime = stoptime - startTime
        print "loop time ", totaltime

    #repeat everything with cluster size of 250

    """

    print "starting everything for cluster of size 250"

    review_1= []
    for i in xrange(len(corpus)):
        review_1 += review_to_sentences(corpus[i], tokenizer,remove_stopwords=True)

    random.shuffle(review_1) #shuffle the corpus

    #print "builing Word2Vec model..."
    model = Word2Vec(review_1, size=num_features, window=15, workers=1)

    centroids = []
    x = 0


    for word in words:

        if x == 0:
            try:
                wordArray = np.asarray(model[word])
                x = 1
            except:
                x = 0
                print word

        else:

            try:
                wordArray = np.vstack((wordArray,[np.asarray(model[word])]))
            except:
                nsdfks = 0

    nclusters = 250
    cluster = KMeans(init='k-means++', n_clusters=nclusters)

    cluster.fit_transform(wordArray)
    cluster_centers = cluster.cluster_centers_
    centroids.append((1,cluster_centers))

    dist = np.linalg.norm(cluster_centers[0] - cluster_centers[1])




    distanceArray = []

    #for center in cluster_centers:

    #for bin size, just find the closest centriod, it goes into that bin, increment that centroid by 1 and use the vector of centroids to classify
    clean_reviews_Pos = []
    clean_reviews_Pos_BOW = []
    clean_reviews_Neg = []
    clean_reviews_Neg_BOW = []
    clean_reviews_Neg_1 = []
    clean_reviews_Pos_1 = []
    print "cleaning and parsing data.."
    #print review_to_words(positiveReviews[1])
    review_size = 0
    for i in xrange(len(positiveReviews)):
        review_1 = review_to_wordlist(positiveReviews[i])  ###################################################################### change way words are used
        #clean_reviews_Pos_BOW.append(review_to_words(positiveReviews[i]))
        clean_reviews_Pos_1.append(review_1)

        review_ = review_to_wordlist(positiveReviews[i])
        clean_reviews_Pos.append(review_)

        review_size += len(review_)
        #print "review size", len(review_)

    for i in xrange(len(negativeReviews)):
        review_1 = review_to_wordlist(negativeReviews[i])  ###################################################################### change way words are used
        clean_reviews_Neg_BOW.append((review_to_words(negativeReviews[i])))
        clean_reviews_Neg_1.append(review_1)

        review_ = review_to_wordlist(negativeReviews[i])
        clean_reviews_Neg.append(review_)

        review_size += len(review_)

    totalsize = len(clean_reviews_Pos) + len(clean_reviews_Neg)


    test_Data = []

    for i in clean_reviews_Pos:
        test_Data.append((1,i))

    for i in clean_reviews_Neg:
        test_Data.append((0,i))

    print test_Data[0]

    allDataArrays = []

    print "starting all data arrays for size 250"

    startTime = time.time()


    for f in xrange(len(test_Data)):
        review = test_Data[f][1]
        label = test_Data[f][0]
        distanceArray = np.zeros(nclusters)
        #print len(review)
        for word in review:
            word = str(word)
            try:
                wordVec = model[word]
            except:
                t = 4


            distArr = []
            for clusterC in cluster_centers:
                distArr.append(np.linalg.norm(wordVec - clusterC))
            #print "distArr", distArr
            #print "distance min",np.argmin(distArr)
            distanceArray[np.argmin(distArr)] += 1
            #print distanceArray
        #print distanceArray
        allDataArrays.append((label,distanceArray))




    print "done with all data arrays for size 250"

    stoptime = time.time()
    totaltime = stoptime - startTime
    print "it took ", totaltime, "seconds"

    posFeatures = getAvgFeatureVecs(clean_reviews_Pos_1, model, num_features)
    negFeatures = getAvgFeatureVecs(clean_reviews_Neg_1, model, num_features)




    #add label to review array
    posFeatures = np.insert(posFeatures,0,1,axis=1)#w2v
    negFeatures = np.insert(negFeatures,0,0,axis=1)#w2v

    #stack training vectors
    allData = np.vstack((posFeatures, negFeatures))#w2v
    labels = allData[:,0]
    allData = np.delete(allData, 0, axis=0)


    #shuffle data
    random.shuffle(test_Data)


    m = len(allDataArrays)
    print m
    k = 3
    crossValNum = 5
    folds = 4

    aucMatrix = np.zeros([k,folds])

    kf = cross_validation.KFold(m,n_folds = folds)
    fold = 0
    method = 2

    print "starting cross val loop for 250"
    startTime = time.time()

    for train_index, test_index in kf:
        fold +=1

        train = list(allDataArrays[f] for f in train_index)
        test = list(allDataArrays[f] for f in test_index)


        trainVectors = np.asarray(list(train[f][1] for f in xrange(len(train))))
        trainLables = np.asarray(list(train[f][0] for f in xrange(len(train))))

        testVectors = np.asarray(list(test[f][1] for f in xrange(len(test))))
        testLables = np.asarray(list(test[f][0] for f in xrange(len(test))))

        predicted = []
        true = []



        #trainVectors = getAvgFeatureVecs(trainVectors, model, num_features)
        #testVectors = getAvgFeatureVecs(testVectors, model, num_features)


        predicted = []
        true = []

        forest_size = 100
        forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
        rcs = 125
        newList = []
        for i in xrange(len(trainVectors)):
            distanceArray = np.zeros(nclusters)

            review = trainVectors[i]
            Tuplabel = trainLables[i]

            for word in review:
                word = str(word)
                try:
                    wordVec = model[word]
                except:
                    break
                distArr = []
                for clusterC in cluster_centers:
                    distArr.append(np.linalg.norm(wordVec - clusterC))

                distanceArray[np.argmin(distArr)] = distanceArray[np.argmin(distArr)] + 1

            newList.append((Tuplabel,distanceArray))


        trainVectors = np.asarray(list(newList[f][1] for f in xrange(len(newList))))
        trainLables = np.asarray(list(newList[f][0] for f in xrange(len(newList))))
        #trainVectors = getAvgFeatureVecs(trainVectors, model, num_features)


        forest_size = 100
        forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
        print trainVectors[0]
        forest = forest.fit(trainVectors,trainLables)
        oob_score =  forest.oob_score_
        print "oob score ", oob_score


        rcsList = [50,75,100,125,150,200]
        count = 0
        scoreKeeper = 0
        for i in xrange(len(testVectors)):


            distanceArray = np.zeros(nclusters)

            review = testVectors[i]
            Tuplabel = testLables[i]

            try:
                preditctedLabel = forest.predict(testVectors)
                count += 1
            except:
                print "fuck"
                g = False

            if preditctedLabel == Tuplabel:
                scoreKeeper+=1


            else:
                if Tuplabel == 1:
                    predicted.append(0)
                else:
                    predicted.append(1)

            true.append(Tuplabel)

        print "fold", fold
        print "scorekeeper",scoreKeeper
        print "count fold", count

        try:
            true = np.asarray(true)
            predicted = np.asarray(predicted)

            print "auc", roc_auc_score(true,predicted)
        except:
            print "auc didnt work"

        stoptime = time.time()
        totaltime = stoptime - startTime
        print "time for loop", totaltime

    """