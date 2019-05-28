#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:56:28 2019

@author: lukasmalik
"""

# ===========================================================================
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

from scripts.load_data import * 
from scripts.load_model import * 
from tqdm import tqdm

df = load_data() 
model = load_model() # loads pretrained word2vec model with 400 dimensions
# ===========================================================================
# calculate the document vector as average of all the words 
def avg_feature_vector(tweet, model, num_features,index2word_set,tokenizer):
    '''
    calculates the average vector 
    '''
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    words = tokenizer.tokenize(tweet)
    for word in words:
        #print(word) # sanity check
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return(feature_vec)

def get_avg_doc2vec(tweets, model, num_features): 
    '''
    calculates the average vector as a representation for a document 
    num_features: number of dimensions of vector 
    '''
    print('Averaging Word Vectors ...')
    tokenizer = TweetTokenizer()
    
    index2word_set = set(model.index2word)
    doc_vecs = []

    for tweet in tqdm(tweets): 
        doc_vec = avg_feature_vector(tweet, model, num_features,index2word_set,tokenizer)
        doc_vecs.append(doc_vec)
    print('... Done')
    
    return(doc_vecs)

doc_vecs = get_avg_doc2vec(df.full_text,model,400)
del model 
del df
#df['doc_vectors'] = get_avg_doc2vec(df['text'],model,400)
# safe new dataframe as pickle 
#df['doc_vectors'] = doc_vecs

#doc_vec_array = df[['id', 'date', 'doc_vectors']].values

#np.save('../data/processed/vectors.npy', doc_vecs)

doc_vecs = np.load('../data/processed/vectors.npy')

df['vecs'] = doc_vecs
# ===========================================================================
#test 

pd.read_pickle("../data/processed/df_vectors.pkl", compression="gzip")

