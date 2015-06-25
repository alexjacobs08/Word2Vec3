import random
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import re
from bs4 import BeautifulSoup
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
imp = Imputer(missing_values="NaN")
np.set_printoptions(threshold=np.nan)

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



#fname = "w2vModelsize300"
num_features = 2500
#model = Word2Vec.load(fname)

positiveDataFile = "../../../../txt_sentoken/pos"
negativeDataFile = "../../../../txt_sentoken/neg"

positiveReviews = []
negativeReviews = []
for i in os.listdir(positiveDataFile):
    with open(os.path.join(positiveDataFile,i),"rb") as f:
        positiveReviews.append(f.read())
    f.close()

for i in os.listdir(negativeDataFile):
    with open(os.path.join(negativeDataFile,i),"rb") as f:
        negativeReviews.append(f.read())
    f.close()

corpus = positiveReviews + negativeReviews
random.shuffle(corpus)
print "builing Word2Vec model..."
model = Word2Vec(corpus, size=num_features, window=50, workers=8)


clean_reviews_Pos = []
clean_reviews_Neg = []
print "cleaning and parsing data.."

for i in xrange(len(positiveReviews)):
    clean_reviews_Pos.append((review_to_words(positiveReviews[i])))

for i in xrange(len(negativeReviews)):
    clean_reviews_Neg.append((review_to_words(negativeReviews[i])))

#model = Word2Vec.load(fname)
#print "loaded w2v model..."

#vectListPos = getCleanReviews(positiveReviews)
#vectListNeg = getCleanReviews(negativeReviews)


#print clean_test_review[1]

print "creating features vectors..."
posFeatures = getAvgFeatureVecs(clean_reviews_Pos, model, num_features)
negFeatures = getAvgFeatureVecs(clean_reviews_Neg, model, num_features)





posFeatures = np.insert(posFeatures,0,1,axis=1)#w2v
negFeatures = np.insert(negFeatures,0,0,axis=1)#w2v




m,n = posFeatures.shape

train = int(.8*m)




trainingData = np.vstack((posFeatures[0:train,:], negFeatures[0:train,:]))#w2v
BOW_trainingData_unShuffled = clean_reviews_Pos[0:train] + clean_reviews_Neg[0:train]#BOW


BOW_train_labels_unShuffled = np.zeros((1,len(BOW_trainingData_unShuffled)),dtype="float64")

BOW_train_labels_unShuffled[0,0:train] = int(1)
BOW_train_labels_unShuffled[0,train:] = int(0)



BOW_trainingData = []
BOW_train_labels = []
index_shuffle = range(len(BOW_trainingData_unShuffled))
random.shuffle(index_shuffle)
for i in index_shuffle:
    BOW_trainingData.append(BOW_trainingData_unShuffled[i])
    BOW_train_labels.append(BOW_train_labels_unShuffled[0][i])




testingData = np.vstack((posFeatures[train+1:,:], negFeatures[train+1:,:]))#w2v
BOW_testingData_unShuffled = clean_reviews_Pos[train+1:] + clean_reviews_Neg[train+1:]#BOW


BOW_test_labels_unShuffled = np.zeros((1,len(BOW_testingData_unShuffled)))

BOW_test_labels_unShuffled[0,0:len(posFeatures)-train] = [1]
BOW_test_labels_unShuffled[0,len(posFeatures)-train:] = [0]

BOW_testingData = []
BOW_test_labels = []
index_shuffle = range(len(BOW_testingData_unShuffled))
random.shuffle(index_shuffle)
for i in index_shuffle:
    BOW_testingData.append(BOW_testingData_unShuffled[i])
    BOW_test_labels.append(BOW_test_labels_unShuffled[0][i])



np.random.shuffle(trainingData)
np.random.shuffle(testingData)

train_labels = trainingData[:,0]
test_labels  = testingData[:,0]

trainingData = np.delete(trainingData,0,axis=1)
testingData = np.delete(testingData,0,axis=1)

where_are_NaNs = np.isnan(trainingData)
trainingData[where_are_NaNs] = 0


where_are_NaNs = np.isnan(testingData)
testingData[where_are_NaNs] = 0

#trainingData = imp.fit_transform(trainingData)
#testingData = imp.fit_transform(testingData)


print "creating bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                         tokenizer = None,    \
                         preprocessor = None, \
                         stop_words = None,   \
                         max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
#print BOW_trainingData
print type(BOW_trainingData[0])
td = getCleanReviews(BOW_trainingData)

BOW_train_data_features = vectorizer.fit_transform(BOW_trainingData).toarray()

# Numpy arrays are easy to work with, so convert the result to an
# array

# ******* Train a random forest using the bag of words
print "training random forest for BOW..."
forest = RandomForestClassifier(n_estimators=1000)

forest.fit(BOW_train_data_features, BOW_train_labels)

BOW_test_data_features = vectorizer.transform(BOW_testingData).toarray()


scoreBOW = forest.score(BOW_test_data_features, BOW_test_labels)

print "score for BOW random forest ", scoreBOW

forest = RandomForestClassifier( n_estimators = 1000 )

print "Fitting a random forest to word vectors size ", num_features," ..."

forest = forest.fit(trainingData,train_labels)

# Test & extract results
#result = forest.predict( llist[2201:,:])

score = forest.score(testingData,test_labels)

print "random forest score ", score

clf = svm.SVC()
clf.fit(trainingData,train_labels)


score2 = clf.score(testingData,test_labels)

print "svm score ", score2

gnb = GaussianNB()

y_preditc = gnb.fit(trainingData,train_labels)
score3 = gnb.score(testingData,test_labels)

print "baesian classifier ", score3





"""

#vectList = [meanVector,gMeanVector,medianVector,minVector,maxVector,stdVector]
vectListNames = ['meanVector','gMeanVector','medianVector','minVector','maxVector','stdVector']
#label = labelArray

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