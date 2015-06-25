
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

positiveDataFile = "../../../../txt_sentoken/pos"
negativeDataFile = "../../../../txt_sentoken/neg"

sentences = []
for i in os.listdir(positiveDataFile):
    with open(os.path.join(positiveDataFile,i),"rb") as f:
        sentences.extend(f.readlines())
    f.close()

for i in os.listdir(negativeDataFile):
    with open(os.path.join(negativeDataFile,i),"rb") as f:
        sentences.extend(f.readlines())
    f.close()

t = open("sentencesPoldata.txt","r+")

t.write(str(sentences))

t.close()

model = Word2Vec(sentences)

model.save("polDataModelDefault")