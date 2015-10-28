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
#print words[0:10]
x=0
wordList = []#will contain all words with wordcoutns above the avg count (somewhere around 30)
for word in words:

    x+=1
    #if x < 10:
        #print word, WordCountsDict[word]
    if int(WordCountsDict[word]) > int(avg):
        #if x < 10:
            #print WordCountsDict[word]
        wordList.append(word)

negWordCountMatrix = np.zeros((len(wordList),len(negativeReviews)))#initalize matrixs to store word counts
posWordCountMatrix = np.zeros((len(wordList),len(positiveReviews)))
revCount = 0#keep track of which review we on

for review in negReviewToList:
    for w in review:
        try:
            ind = wordList.index(w)
            negWordCountMatrix[ind,revCount] += 1

        except ValueError:
            x = 3

    revCount+=1

revCount = 0
for review in posReviewToList:
    for w in review:
        try:
            ind = wordList.index(w)
            posWordCountMatrix[ind,revCount] += 1

        except ValueError:
            x = 3

    revCount+=1


posSum = np.sum(posWordCountMatrix,axis=1)
negSum = np.sum(negWordCountMatrix,axis=1)

sumDif = (posSum - negSum)


posAvg = np.mean(posWordCountMatrix,axis=1)
negAvg = np.mean(negWordCountMatrix,axis=1)

avgDif = (posAvg-negAvg)

posStd = np.std(posWordCountMatrix,axis=1)
negStd = np.std(negWordCountMatrix,axis=1)

nWords = 30

"""nLargest = heapq.nlargest(nWords, enumerate(avgDif), key=lambda x: x[0])
nLarList = np.asarray([nLargest[i][0] for i in xrange(len(nLargest))],int)
nLarWordList = []
for n in nLarList:
    nLarWordList.append(wordList[n])

nSmallest = heapq.nsmallest(nWords, enumerate(avgDif), key=lambda x: x[1])
nSmalList = np.asarray([nSmallest[i][0] for i in xrange(len(nSmallest))],int)
nSmalWordList = []
for n in nSmalList:
    nSmalWordList.append(wordList[n])

print nLarWordList
print "    "
print nSmalWordList


wc = 0 #word counter
posSumNorm = normalize(posSum)
negSumNorm = normalize(negSum)
zList = []
for word in wordList:
    x1 = posSum[wc]
    x2 = negSum[wc]
    n1 = 1000
    n2 = 1000
    p1 = x1/n1
    p2 = x1/n2
    Pp = (x1+x2)/(n1+n2)

    z = (p1-p2) / ( math.sqrt( Pp * (1-Pp) ) * math.sqrt( (1/n1) + (1/n2) ) )
    zList.append(z)
    wc+=1

nSmallest = heapq.nsmallest(nWords, enumerate(zList), key=lambda x: x[1])
nSmalList = np.asarray([nSmallest[i][0] for i in xrange(len(nSmallest))],int)

zWordList = []
for n in nSmalList:
    zWordList.append(wordList[n])

print zWordList

"""
#print posWordCountMatrix.shape

Pvales = []


for i in xrange(posWordCountMatrix.shape[0]):

    Pvales.append(scipy.stats.ttest_ind(posWordCountMatrix[i,:],negWordCountMatrix[i,:])[1])


newMat = np.hstack((posWordCountMatrix,negWordCountMatrix))

var = list(scipy.var(newMat,axis=1))
print len(var)
print var[0:10]


review_1= []
for i in xrange(len(corpus)):
    review_1 += review_to_sentences(corpus[i], tokenizer,remove_stopwords=True)

random.shuffle(review_1) #shuffle the corpus

#print "builing Word2Vec model..."
model = Word2Vec(review_1, size=num_features, window=15, workers=1)

nEnumList = [400]

for nN in nEnumList:
    #Pvales = heapq.nsmallest(nN, enumerate(Pvales), key=lambda x: x[1]) for Pvalue approach
    #pList = np.asarray([Pvales[i][0] for i in xrange(len(Pvales))],int)

    Pvales = heapq.nlargest(nN, var) #for variance approach
    print Pvales[0:10]
    pList = np.asarray([Pvales[i] for i in xrange(len(Pvales))],int)

    nWordList = []
    for n in pList:
        nWordList.append(wordList[n])

    print nWordList

    #print nWordList[0:300]

    #fname = "300features_40minwords_10context"
    #num_features = 300
    #model = Word2Vec.load(fname)

    review_1= []
    for i in xrange(len(corpus)):
        review_1 += review_to_sentences(corpus[i], tokenizer,remove_stopwords=True)

    random.shuffle(review_1) #shuffle the corpus

    #print "builing Word2Vec model..."
    model = Word2Vec(review_1, size=num_features, window=15, workers=1)

    #go through all reviews and cluster into k clusters using k means based on the vectors from the n important words


    centroids = []
    for review in positiveReviews:
        review1 = review_to_wordlist(review)

        x = 0
        for word in review1:
            if x == 0:
                try:
                    reviewArray = np.asarray(model[word])
                    x=1
                except:
                    x = 0
                    #print word

            else:
                #print "else"
                if word in nWordList:
                    #print "in word list"
                    #vect = makeFeatureVec(word,model,num_features)
                    try:
                        reviewArray = np.vstack((reviewArray,[np.asarray(model[word])]))
                    except:
                        nsdfks = 0

        try:
            cluster = KMeans(init='k-means++', n_clusters=50)
            cluster.fit_transform(reviewArray)
            cluster_centers = cluster.cluster_centers_
            centroids.append((1,cluster_centers))
        except:
            nNTnnt = 9

    for review in negativeReviews:
        review1 = review_to_wordlist(review)

        x = 0
        for word in review1:
            if x == 0:
                try:
                    reviewArray = np.asarray(model[word])
                    x=1
                except:
                    x = 0
                    #print word

            else:
                #print word
                if word in nWordList:
                    #print "in word list"
                    #vect = makeFeatureVec(word,model,num_features)
                    try:
                        reviewArray = np.vstack((reviewArray,[np.asarray(model[word])]))
                    except:
                        nntnt = 0
        try:
            cluster = KMeans(init='k-means++', n_clusters=50)
            cluster.fit_transform(reviewArray)
            cluster_centers = cluster.cluster_centers_
            centroids.append((0,cluster_centers))
        except:
           nnTnnt = 0



    print "shuffling centroids"
    random.shuffle(centroids)
    m = len(centroids)

    print "m"

    folds = 4


    kf = cross_validation.KFold(m,n_folds = folds)

    scores = []
    for train_index, test_index in kf:

        train = list(centroids[f] for f in train_index)
        test = list(centroids[f] for f in test_index)


        trainVectors = np.asarray(list(np.ravel(train[f][1])for f in xrange(len(train))))
        trainLables = np.asarray(list(train[f][0] for f in xrange(len(train))))



        testVectors = np.asarray(list(np.ravel(test[f][1])for f in xrange(len(test))))
        testLables = np.asarray(list(test[f][0] for f in xrange(len(test))))


        forest_size = 1000
        forest = RandomForestClassifier( n_estimators = forest_size )
        forest = forest.fit(trainVectors,trainLables)
        scores.append(forest.score(testVectors,testLables))

    score = sum(scores)/float(len(scores))
    print "score with ", nN, " important words is ", score



