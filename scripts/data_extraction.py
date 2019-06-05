# imports 
import pandas as pd 
import numpy as np 
import re 
import glob 
import os 
from tqdm import tqdm

def extract_jsons(path,is_control_int): 
    os.chdir('/Users/lukasmalik/Desktop/Praktikum CSH/project-internship/')
    timelines = glob.glob(os.path.join(path, "*.json"))
    dataframe = pd.DataFrame()
    
    for i in tqdm(range(0,len(timelines))):
        data = pd.read_json(timelines[i], lines=True)
        # check
        if len(data.index) != 0:
            data = data[data['in_reply_to_user_id'].isnull()]
            try:
                data = data[data['retweeted_status'].isnull()]
            except: 
                pass
            data = data[data['retweeted'] == False]
            data = data[data['lang'] == 'en']
            user_id = re.findall(r'\d+',timelines[i])[0]
            data['id'] = user_id
            data['is_control'] = is_control_int
            data = data[['id','is_control','created_at','full_text']]
            dataframe = dataframe.append(data, ignore_index=True)
            
    if is_control_int == 1: 
        dataframe.to_pickle("./data/processed/control_data1.pkl")
    else:
        dataframe.to_pickle("./data/processed/bipolar_data1.pkl")
    return(dataframe)

path = "data/raw/Timelines/"
dataframe = extract_jsons(path,1)
