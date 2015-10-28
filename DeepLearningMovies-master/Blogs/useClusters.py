#import things needed to process text
#ipmort txt files
#import cluster list

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

#shorten for testing

positiveReviews = positiveReviews[0:500]
negativeReviews = negativeReviews[0:500]


posReviewToList = []
negReviewToList = []

BOW_pos = []
BOW_neg = []

for i in xrange(len(positiveReviews)):
    review = review_to_wordlist(positiveReviews[i], remove_stopwords= True)

    posReviewToList.append(review)

    BOW_pos.append(" ".join(review))

for i in xrange(len(negativeReviews)):
    review = review_to_wordlist(negativeReviews[i],remove_stopwords= True)

    negReviewToList.append(review)

    BOW_neg.append(" ".join(review))

print "running"

###########################################################################
###########################################################################
#BOW
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# BOW_pos = []
# for i in xrange(len(posReviewToList)):
#     BOW_pos.append((posReviewToList[i],1))
#
# BOW_neg = []
# for i in xrange(len(negReviewToList)):
#     BOW_neg.append((posReviewToList[i],0))
#
print "starting BOW vectorizaiton"

BOW_all = BOW_pos + BOW_neg

BOW_features = vectorizer.fit_transform(BOW_all)
BOW_features = BOW_features.toarray()



#BOW_neg_features = vectorizer.fit_transform(BOW_neg)
#BOW_neg_features = BOW_neg_features.toarray()

print len(BOW_features)

BOW_Data = []
for i in xrange(len(BOW_pos)):
    BOW_Data.append((BOW_features[i],1))

BOW_negAp = []
for i in xrange(len(BOW_pos),len(BOW_pos) + len(BOW_neg)):
    BOW_Data.append((BOW_features[i],0))

print len(BOW_Data)
random.shuffle(BOW_Data)
###########################################################################
###########################################################################


#read through each review, increase count of clusterArray thing


clusterFile = "../../../../trunk/classes.sorted.txt"

words = []
clusters = []
with open(clusterFile) as f:

    line = f.readlines()
    f.close()

for line in line:
    word,clustNum = str.split(line)

    words.append(word)
    clusters.append(int(clustNum))

startTime = time.time()

print "clustering positive"
clusterArraysPostitive = []
for review in posReviewToList:
    clusterArray = np.zeros(500)
    for word in review:
        try:
            clusterNumber = clusters[words.index(word)]
            clusterArray[clusterNumber] += 1
        except:
            x = 0
    clusterArraysPostitive.append((clusterArray,1))

stoptime = time.time()
totaltime = stoptime - startTime
print "time for cluster positive ", totaltime

startTime = time.time()

print "clustering negative"
clusterArraysNegative = []
for review in negReviewToList:
    clusterArray = np.zeros(500)
    for word in review:
        try:
            clusterNumber = clusters[words.index(word)]
            clusterArray[clusterNumber] += 1
        except:
            x = 0
    clusterArraysNegative.append((clusterArray,0))

stoptime = time.time()
totaltime = stoptime - startTime
print "time for cluster neg ", totaltime



testData = clusterArraysPostitive + clusterArraysNegative

random.shuffle(testData)


m = len(testData)
folds = 4
fold = 0

kf = cross_validation.KFold(m,n_folds = folds)

print "starting cross vald loop"

startTime = time.time()

scores = []
BOW_scores = []
for train_index, test_index in kf:
    fold +=1

    train = list(testData[f] for f in train_index)

    test = list(testData[f] for f in test_index)


    trainVectors = np.asarray(list(train[f][0] for f in xrange(len(train))))
    trainLables = np.asarray(list(train[f][1] for f in xrange(len(train))))

    print trainVectors.shape
    print trainLables.shape


    testVectors = np.asarray(list(test[f][0] for f in xrange(len(test))))
    testLables = np.asarray(list(test[f][1] for f in xrange(len(test))))

    predicted = []
    true = []

    forest_size = 100
    forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
    #print trainVectors[0]
    forest = forest.fit(trainVectors,trainLables)
    oob_score = forest.oob_score_
    print "oob score ", oob_score

    score = forest.score(testVectors, testLables)
    scores.append(score)
    print "score ", score, " for fold ", fold

    stoptime = time.time()
    totaltime = stoptime - startTime
    print "loop time ", totaltime


    ###########################################################################
    ###########################################################################
    print len(BOW_Data)
    print BOW_Data[0]
    BOW_train = list(BOW_Data[f] for f in train_index)

    BOW_test = list(BOW_Data[f] for f in test_index)


    BOW_trainVectors = np.asarray(list(BOW_train[f][0] for f in xrange(len(BOW_train))))
    BOW_trainLables = np.asarray(list(BOW_train[f][1] for f in xrange(len(BOW_train))))


    BOW_testVectors = np.asarray(list(BOW_test[f][0] for f in xrange(len(BOW_test))))
    BOW_testLables = np.asarray(list(BOW_test[f][1] for f in xrange(len(BOW_test))))

    print BOW_trainVectors.shape
    print BOW_trainLables.shape
    forest_size = 100
    forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
    #print trainVectors[0]
    forest = forest.fit(BOW_trainVectors,BOW_trainLables)
    oob_score = forest.oob_score_
    print "BOW oob score ", oob_score


    BOW_score = forest.score(BOW_testVectors, BOW_testLables)
    BOW_scores.append(BOW_score)
    print "BOW score ", score, " for fold ", fold

    stoptime = time.time()
    totaltime = stoptime - startTime
    print "BOW loop time ", totaltime





print "average score w2v", (sum(scores)/float(len(scores)))
print "average score BOW", (sum(BOW_scores)/float(len(BOW_scores)))




#keep each clusterArray with the review
#classify the reviews
