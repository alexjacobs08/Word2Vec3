
import random
import os
from nltk.corpus import stopwords
import nltk.data
#nltk.download()
import re
from bs4 import BeautifulSoup
import numpy as np
import scipy
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import cross_validation
imp = Imputer(missing_values="NaN")

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
positiveDataFile = "../../../../../txt_sentoken/pos"
negativeDataFile = "../../../../../txt_sentoken/neg"

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



review_1 = []
for i in xrange(len(corpus)):
    review_1 += review_to_sentences(corpus[i],tokenizer)





review_1 = [' '.join(x) for x in review_1]
#print review_1[0:100]
words = re.findall(r'\w+', str(review_1))
print words[0:100]






cap_words = [word.upper() for word in words]
#print cap_words[0:100]
word_counts = list(Counter(cap_words).values())
words = list(Counter(cap_words).keys())
print len(word_counts)
print len(words)
newList =  heapq.nlargest(200,word_counts)
topWords = []
for t in xrange(len(newList)):
    topWords.append(words[word_counts.index(newList[t])])
    print words[word_counts.index((newList[t]))]




review_1 = []

for i in xrange(len(corpus)):
    review_1 += review_to_sentences(corpus[i], tokenizer,remove_stopwords=False)

random.shuffle(review_1) #shuffle the corpus

print "builing Word2Vec model..."
model = Word2Vec(review_1, size=num_features, window=15, workers=1)
#print "loading word2vec model..."
#model = Word2Vec.load_word2vec_format('../../../../../Data/GoogleNews-vectors-negative300.bin', binary=True)


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

print "average words", review_size/totalsize

#model = Word2Vec.load(fname)
#print "loaded w2v model..."


"""
posFeatures = getAvgFeatureVecs(clean_reviews_Pos_1, model, num_features)
negFeatures = getAvgFeatureVecs(clean_reviews_Neg_1, model, num_features)


m,n = posFeatures.shape

#determine lenght of training set
train = int(.8*m)

#append BOW lists together for training data
BOW_trainingData_unShuffled = clean_reviews_Pos_BOW[0:train] + clean_reviews_Neg_BOW[0:train]#BOW

#create BOW lables
BOW_train_labels_unShuffled = np.zeros((1,len(BOW_trainingData_unShuffled)),dtype="float64")
BOW_train_labels_unShuffled[0,0:train] = int(1)
BOW_train_labels_unShuffled[0,train:] = int(0)


#shuffle BOW training list and lables to match indexes
BOW_trainingData = []
BOW_train_labels = []
index_shuffle = range(len(BOW_trainingData_unShuffled))
random.shuffle(index_shuffle)
for i in index_shuffle:
    BOW_trainingData.append(BOW_trainingData_unShuffled[i])
    BOW_train_labels.append(BOW_train_labels_unShuffled[0][i])


BOW_testingData_unShuffled = clean_reviews_Pos_BOW[train+1:] + clean_reviews_Neg_BOW[train+1:]#BOW

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
#
# Numpy arrays are easy to work with, so convert the result to an array
BOW_train_data_features = vectorizer.fit_transform(BOW_trainingData).toarray()
BOW_test_data_features = vectorizer.transform(BOW_testingData).toarray()

######## Train a random forest using the bag of words
print "training random forest for BOW..."
forest_size = 100
forest = RandomForestClassifier(n_estimators= forest_size)
forest.fit(BOW_train_data_features, BOW_train_labels)
scoreBOW1 = forest.score(BOW_test_data_features, BOW_test_labels)
print "score for BOW random forest with ", forest_size," estimators ", scoreBOW1

######## Train and test svm for BOW

clf = svm.SVC()
clf.fit(BOW_train_data_features, BOW_train_labels)
scoreBOW2 = clf.score(BOW_test_data_features, BOW_test_labels)
print "score for BOW svm ", scoreBOW2
######## Train and test baesian classifier for BOW

gnb = GaussianNB()
y_preditc = gnb.fit(BOW_train_data_features, BOW_train_labels)
scoreBOW3 = gnb.score(BOW_test_data_features, BOW_test_labels)

print "score for BOW baesian classifier ", scoreBOW3

#repeat with word vectors
print "Fitting a random forest to word vectors size ", num_features," ..."



print "creating features vectors..."
#create array of vectors for each array
"""

test_Data = []

for i in clean_reviews_Pos:
    test_Data.append((1,i))

for i in clean_reviews_Neg:
    test_Data.append((0,i))

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


m,n = allData.shape

k = 3
crossValNum = 5
folds = 4

aucMatrix = np.zeros([k,folds])

kf = cross_validation.KFold(m,n_folds = folds)
fold = 0
method = 2

for train_index, test_index in kf:
    fold +=1

    train = list(test_Data[f] for f in train_index)
    test = list(test_Data[f] for f in test_index)

    trainVectors = np.asarray(list(train[f][1] for f in xrange(len(train))))
    trainLables = np.asarray(list(train[f][0] for f in xrange(len(train))))

    testVectors = np.asarray(list(test[f][1] for f in xrange(len(test))))
    testLables = np.asarray(list(test[f][0] for f in xrange(len(test))))


    #trainVectors = getAvgFeatureVecs(trainVectors, model, num_features)
    #testVectors = getAvgFeatureVecs(testVectors, model, num_features)


    predicted = []
    true = []

    forest_size = 100
    forest = RandomForestClassifier( n_estimators = forest_size, oob_score=True )
    rcs = 125
    newList = []
    for i in xrange(len(trainVectors)):

        review = trainVectors[i]
        Tuplabel = trainLables[i]

        length = len(review)
        size = length / rcs
        remainder = length % rcs

        #scoreKeeper = 0

        n = 0
        k = 0
        #count = 0

        for j in xrange(size):
            n = j * rcs
            k = n + rcs
            if k > length:
                k = remainder
            review_cut = review[n:k]
            #review_Array = np.asarray(makeFeatureVec(review_cut, model, num_features))
            newList.append((Tuplabel,review_cut))


    trainVectors = np.asarray(list(newList[f][1] for f in xrange(len(newList))))
    trainLables = np.asarray(list(newList[f][0] for f in xrange(len(newList))))
    trainVectors = getAvgFeatureVecs(trainVectors, model, num_features)

    forest = forest.fit(trainVectors,trainLables)
    oob_score =  forest.oob_score_
    print "oob score ", oob_score


    rcsList = [50,75,100,125,150,200]

    for i in xrange(len(testVectors)):


        g = True
        review = testVectors[i]
        Tuplabel = testLables[i]

        length = len(review)
        size = length / rcs
        remainder = length % rcs

        scoreKeeper = 0


        if method == 1:

            n = 0
            k = 0
            count = 0
            for j in xrange(size):
                n = j * rcs
                k = n + rcs
                if k > length:
                    k = remainder
                review_cut = review[n:k]
                review_Array = np.asarray(makeFeatureVec(review_cut, model, num_features))

                try:
                    preditctedLabel = forest.predict(review_Array)[0]
                    count += 1
                except:
                    print "fuck"
                    g = False
                    break

                if preditctedLabel == Tuplabel:
                    scoreKeeper+=1

                if scoreKeeper > count*.5 and g == True:
                    correct +=1
                    predicted.append(Tuplabel)

                else:
                    if Tuplabel == 1:
                        predicted.append(0)
                    else:
                        predicted.append(1)

                true.append(Tuplabel)



        elif method == 2:
            wordList = words[0:100]#needs to be wordslist
            countz = 0
            window = 2
            wrCount = 0
            reviewSum = 0
            for word in review:

                wrCount += 1

                if word in wordList:
                    wordSum = 0
                    for wPlus in window:

                        wordSum += makeFeatureVec(review[countz-wPlus],model,num_features)

                    for wMinus in window:

                        wordSum += makeFeatureVec(review[countz+wPlus],model,num_features)

                    wordSum = float(wordSum) / float(window*2)

                reviewSum += wordSum
            reviewTotal = reviewSum / wrCount


            predictedLable = forest.predict(reviewTotal)

            if predictedLable == Tuplabel:
                correct +=1

        score = float(correct) / float((len(testVectors)))

        if method ==1:

            print "for rcs of ",rcs,"fold number ", fold,"and score of ", score
        elif method == 2:
            print "for method 2 and fold number ", fold, " score is ", score


    true = np.asarray(true)
    predicted = np.asarray(predicted)

    print "auc", roc_auc_score(true,predicted)



#print aucMatrix
"""

print "Fitting a random forest to word vectors size ", num_features," ..."
forest_size = 1000
forest = RandomForestClassifier( n_estimators = forest_size )
forest = forest.fit(trainingData,train_labels)
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