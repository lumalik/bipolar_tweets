# imports 
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize 

# ===========================================================================
# read in data 
data_filtered = pd.read_pickle("./data/processed/control_data.pkl") # substitute later 

# ===========================================================================
# tag each word with corresponding document 
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), 
                              tags=[str(i)]) for i, _d in enumerate(data_filtered['text'])]

# ===========================================================================
# save tagged_data as text file for later use 
path = "data/processed/tagged_data.txt"

file = open("tagged_control.txt","w") 
file.write(tagged_data, 'w')
file.close


# ===========================================================================
# create model 
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved") 

# ===========================================================================
# implement model 
from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I hate everything".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])

