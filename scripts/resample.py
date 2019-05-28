#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:29:47 2019

@author: lukasmalik
"""
import os 
os.chdir('/Users/lukasmalik/Desktop/Praktikum CSH/project-internship/scripts')

# imports 
import itertools
import numpy as np 
import pandas as pd 
from scipy.spatial import distance

from load_data import *
# ===========================================================================
df = load_data()
model = load_model()
# ===========================================================================
# calculate cosine similarity    
def avg_cosine_similarity(vectors):
    similarities = []
    # calculate angle 
    #vectors = np.asarray(vectors, dtype=np.float32)
    for word_vec1, word_vec2 in itertools.combinations(vectors, 2):
        sim = 1 - distance.cosine(word_vec1, word_vec2)
        similarities.append(sim)
    return(np.mean(similarities))

# test 
# for each user 
#df.index = df['date']
   
#df[['id','date','vecs']].groupby('id').resample('W', on = 'date')['vecs'].apply(avg_cosine_similarity)
doc_vecs = np.load('../data/processed/vectors.npy')
df['vecs'] = pd.Series(list(doc_vecs))
del doc_vecs

df = df.reset_index().set_index('created_at')
df = df[df.index >= '2014-01-01 00:00:00']
grouper = df.groupby([pd.Grouper(freq='M'),'id'])

# filter series before resample 
#filter_resampled = grouper['vecs'].count()
#filter_resampled[filter_resampled ==1]

## 1000: 00.602118
## 10000: 0:00:26.087830
## 100000: 


series_resampled = grouper['vecs'].apply(avg_cosine_similarity)

# create a dataframe an save it 
df_resampled = pd.DataFrame({'id': series_resampled.index.get_level_values(1),
                             'date':series_resampled.index.get_level_values(0), 
                             'cosine_similarity_monthly':series_resampled.values})

 # save as pickle     
df_resampled.to_pickle("../data/processed/df_resampled_m.pkl")
 
 # read 
test = pd.read_pickle("../data/processed/df_resampled_m.pkl")  


# join with id 
df = df[['id','is_control']].drop_duplicates(subset=['id'])
#
df_result = df_resampled.merge(df, on= 'id', how='left')
#
# 
df_result.to_pickle("../data/processed/df_resampled_m.pkl")
#   
 
# ===========================================================================
# other functions 
def max_cosine_similarity(vectors):
    similarities = []
    # calculate angle 
    vectors = np.asarray(vectors, dtype=float)
    for word_vec1, word_vec2 in itertools.combinations(vectors, 2):
        sim = 1 - distance.cosine(word_vec1, word_vec2)
        similarities.append(sim)
    return(max(sim))
   
# calculate euclidean distance
def max_distance(vectors):
    distances = []
    vectors = np.asarray(vectors, dtype=float)

    # calculate distance for each vector to each other vector 
    for word_vec1, word_vec2 in itertools.combinations(vectors, 2):
        distance = np.linalg.norm(word_vec1 - word_vec2)
        distances.append(distance)
    return(max(distances))

# similarity over time 
def resample_time(df, column, method = 'cosine', time_range='W'):
    df.index = df['date']
    if method == 'cosine': 
        resample = df[column].resample(time_range).avg_cosine_similarity()
    if method == 'euclidean': 
        resample = df[column].resample(time_range).max_distance()
    return(resample)
    
