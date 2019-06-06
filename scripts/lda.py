# topic modeling on tweet datasets
from scripts.load_data import *
df = load_data()

# ===========================================================================
# data preparation 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim import corpora
import pickle

from tqdm import tqdm

nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def get_lemma(word):
    '''
    input: word to lemmatize
    output: lemmatized version of the word 
    example: dogs -> dog 
    '''
    return WordNetLemmatizer().lemmatize(word)
    
def prepare_text_for_lda(tweets):
    '''
    input: expects dataframe column 'full_text' 
    output: corpus, each tweet being a list of tokenized words
    '''
    corpus = []
    tokenizer = TweetTokenizer()
    for tweet in tqdm(tweets): 
        tokens = tokenizer.tokenize(tweet)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in en_stop]
        tokens = [get_lemma(token) for token in tokens]
        corpus.append(tokens)
    return corpus

# ===========================================================================
def create_corpora_dict(texts): 
    text_data = prepare_text_for_lda(texts)
    num_features = 100000 #read somewhere 
    dictionary = corpora.Dictionary(text_data) #, prune_at = num_features
    dictionary.filter_extremes(no_below=100,no_above=0.5, keep_n=num_features)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(text) for text in text_data]
    return dictionary,corpus 

# create and save corpora for both 
(dictionary,corpus) = create_corpora_dict(df.full_text)

pickle.dump(corpus, open('../data/processed/corpus.pkl', 'wb'))
dictionary.save('../data/processed/dictionary.gensim')

# create and save corpora for bipolar 
(dictionary,corpus) = create_corpora_dict(df.full_text[df.is_control == 0])

pickle.dump(corpus, open('../data/processed/corpus_bipolar.pkl', 'wb'))
dictionary.save('../data/processed/dictionary_bipolar.gensim')

# create and save corpora for control 
(dictionary,corpus) = create_corpora_dict(df.full_text[df.is_control == 1])

pickle.dump(corpus, open('../data/processed/corpus_control.pkl', 'wb'))
dictionary.save('../data/processed/dictionary_control.gensim')


# =========================================================================== 
# get lda model 

import gensim
NUM_TOPICS = 10

dictionary = gensim.corpora.Dictionary.load('../data/processed/dictionary.gensim')
corpus = pickle.load(open('../data/processed/corpus.pkl', 'rb'))

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=1)
ldamodel.save('../models/lda.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print("")
    print(topic)
    print("")
    print("_______________________________________________")
 
# =========================================================================== 
# visualizing 
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('../data/processed/corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('lda.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
