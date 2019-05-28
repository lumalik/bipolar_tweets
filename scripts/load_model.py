#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:48:51 2019

@author: lukasmalik
"""

def load_model(pretrained=True):
    import os 
    os.chdir('/Users/lukasmalik/Desktop/Praktikum CSH/project-internship/scripts')
    import datetime
    start_time = datetime.datetime.now()
    if pretrained == True: 
        import os
        import sys
        from word2vecReader import Word2Vec
    
        os.environ['PYTHONINSPECT'] = 'True'
    
        model_path = "../models/word2vec_twitter_model.bin"
        print("Loading the model, this can take some time...")
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: "+str(len(model.vocab)))
        print("--- %s seconds ---" % (datetime.datetime.now() - start_time))
    return(model)
    
model = load_model()


