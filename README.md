There is a good python(or cython) implementation that was ported and added to the gensim package. here's the API for that http://radimrehurek.com/gensim/models/word2vec.html
Additionally, as part of a Kaggle project there are functions written in python that can give you the average vector for a sentence or a paragraph https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

essentially building the model would look like this 
 # Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
(There are actually more possible parameters than the ones used in this model, detailed in the API)

senetences is the entire corpus with simply one sentence per line
num_features is the length of the word vectors
min_word_count is the minimum number of appearances a word must have to be included in the model
context is the maximum distance between the current and predicted word within a sentence
downsampling is for ignoring frequent words (i.e. "and", "the", ect)

once the model is trained it can be saved

accessing an individual word vector within in the model is simple, for example
>>> model["school"]
will return a numpy array of size 1xnum_features (so in the case of our example model, 1x300)

the exciting functions are part of the kaggle project (located in Part 3 under information if you're interested in a more in-depth look)
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
(reviews in this case is specific to the kaggle project, classifying IMBD (movie) reviews with either a positive or negative sentiment)

getAvgFeatureVecs calls makeFeatureVec, looping through all of them and returning a vector for each "review".  In our case a "review" would be a note, and so each note would have a vector associated with it.

From what I understand about how our pipeline runs right now, each patient has the same number of features associated with them despite a variable number of notes.  

I think we have a few options here: 

1. change the pipeline to look at each note as independent of a patient, but either the note identifies as lupus or not lupus by itself
2. find a way to average or reduce all of a patients notes into one vector or into a a set number of vectors
3. merge all of the notes together for each patient
4. .... i'm sure there are more ways I haven't thought of, maybe some combination of the above even

my concern with 1, i have no idea if it makes any sense to do this
my concern with 2, I feel like the more normalization we do, the more semantic meaning we lose.
my concern with 3, these functions were written more for paragraph sized input, I don't know if this would preserve any meaning or if too much data would bloat everything


The really good news is that all of the word2vec functions and additional kaggle functions seem very straight forward; the implementation should be cake, and we can try all three ideas and more. From what I understand this is perfect, exactly what we were looking for, but I'm still understanding how our current pipeline works and what exactly our data looks like so correct me if I said anything that didn't make sense. 

I am still waiting on my netID.  I emailed Candace on Monday for an update and she said the request had been submitted and that she would let me know when it came through.
