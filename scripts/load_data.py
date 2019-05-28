#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:58:58 2019

@author: lukasmalik
"""

def load_data(): 
    '''
    loads bipolar and control data from folder processed 
    returns concatinated dataframe with timestamp index 
    '''
    import pandas as pd 
    
    # read data 
    bipolar_data = pd.read_pickle("../data/processed/bipolar_data.pkl")
    control_data = pd.read_pickle("../data/processed/control_data.pkl")

    # columns are the same check 
    control_data.columns == bipolar_data.columns

    # merge data 
    df = pd.concat([control_data, bipolar_data],ignore_index=True)
    
    # set timestamp as index 
    #df.index = df['date']
    
    return(df)

