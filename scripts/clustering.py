# visualize document vectors in using pca 

# imports 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from load_data import *
# ===========================================================================
# load word vectors
doc_vecs = np.load('../data/processed/vectors.npy')
df = load_data()
# ===========================================================================
# plotting 
    
def vec_plot(vector,size=400): 
    '''
    expects vector 
    format: text | vector | label 
    '''
    
    # scale 
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vector)
    print('scaling done ...')
    
    # pca 
    pca = PCA(n_components=2) 
    transformed = pd.DataFrame(pca.fit_transform(scaled))
    print('pca done ...')
    
    # get id 
    df = load_data()
    transformed['is_control'] = df['is_control']
    print('control tag added ...')
    
    return(transformed)
    

transformed = vec_plot(doc_vecs)

del doc_vecs
#to pickle
#transformed.to_pickle("../data/processed/pca_transformed.pkl")

# read pickle 
transformed = pd.read_pickle("../data/processed/pca_transformed.pkl")  
#transformed['full_text'] = df['full_text']
# take a sample since all tweets are too much to plot 
transformed_sample=transformed.sample(n=10000)

# plot 
plt.scatter(transformed_sample[0],
            transformed_sample[1],
            c=transformed_sample["is_control"],
            alpha=0.5)
plt.legend()
plt.show()
 
    