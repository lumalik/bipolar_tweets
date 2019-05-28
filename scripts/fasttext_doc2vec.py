# make sure you are in right directory 
import os 
os.chdir('/Users/lukasmalik/Desktop/Praktikum CSH/project-internship/')

#doc2vec using fasttext 

# imports  ==================================================
import numpy as np
import pandas as pd
from scipy import spatial
from gensim.models import FastText
import re

# sample tweet and tweets
# ===========================================================================
#sample1 = 'Gym was everything I will feel it in the am'
#sample2 = '@oqDLD8mQxfW0WIU only worry is how long it will last :-/'
#samples = pd.DataFrame({'text':[sample1,sample2]}) # build sample dataframe
# ===========================================================================
# read data 
bipolar_data = pd.read_pickle("./data/processed/bipolar_data.pkl")
control_data = pd.read_pickle("./data/processed/control_data.pkl")

# columns are the same check 
control_data.columns == bipolar_data.columns

# merge data and delete to free memory 
df = pd.concat([control_data, bipolar_data])
del bipolar_data 
del control_data

# set timestamp as index 
df.index = df['date']

# ===========================================================================
# preprocessing for word vectors
def preprocess(tweets): 
    tweet_words = []
    for tweet in tweets: 
        # store as list of words
        #tokens = re.sub(r"[^a-z0-9]+", " ", tweet.lower()).split()
        tokens = str(tweet).lower().split()
        #tweet_words.extend(tokens)
        tweet_words.append(tokens)
    return(tweet_words)
    
tweet_words = preprocess(bipolar_data['text'])
del df
# ===========================================================================
# getting fasttext vectors
#model = FastText(tweet_words, size=100, window=3, min_count=1)
model = FastText(size=100, window=3, min_count=1)
model.build_vocab(tweet_words,update=True)
model.train(tweet_words, total_examples=model.corpus_count, epochs=10)

# ===========================================================================
# save and load model 
from gensim.test.utils import get_tmpfile

fname = get_tmpfile("fasttext.model")
model.save(fname)
model = FastText.load(fname)
# ===========================================================================
# calculate the document vector as average of all the words 
index2word_set = set(model.wv.index2word)

def avg_feature_vector(words, model, num_features, word_set):
    '''
    calculates the average vector 
    '''
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_avg_doc2vec(words, model, num_features, word_set): 
    '''
    calculates the average vector as a representation for a document 
    num_features: number of dimensions of vector 
    '''
    doc_vecs = []
    for doc in tweet_words: 
        doc_vec = avg_feature_vector(doc, model, num_features, index2word_set)
        doc_vecs.append(doc_vec)
    return(doc_vecs)

doc_vecs = get_avg_doc2vec(tweet_words,model,100,index2word_set)
# ===========================================================================
# create dataframe from document vectors 
# format: text | avg_vector | label 

doc_vecs_df = pd.DataFrame({'text':samples['text'],
                           'avg_vector':doc_vecs,
                           'label':'control'})

# save df as pickle 
doc_vecs_df.to_pickle("./data/processed/doc_vecs_df.pkl")
  
# open 
doc_vecs_df = pd.read_pickle("./data/processed/doc_vecs_df.pkl")
   


