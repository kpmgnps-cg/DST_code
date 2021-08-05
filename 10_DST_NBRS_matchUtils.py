import re
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import gzip

# def fitNearestN(data):
#     vectorizer = CountVectorizer(min_df=0, analyzer='word', lowercase=False)
#     tf_idf_matrix = vectorizer.fit_transform(data)
#     nbrs = NearestNeighbors(n_neighbors=3, metric='cosine', leaf_size=30, radius=10, algorithm='brute').fit(tf_idf_matrix)    
#     with gzip.open('./app/models/vectorizer_words.pickle', 'wb') as handle:
#         pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#     with gzip.open('./app/models/nbrs_words.pickle', 'wb') as handle:
#         pickle.dump(nbrs, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#     return nbrs,vectorizer

def fitNearestGrams(data):
  global config
  vectorizer = CountVectorizer(min_df=0, analyzer=ngrams, lowercase=False)
  tf_idf_matrix = vectorizer.fit_transform(data)
  nbrs = NearestNeighbors(n_neighbors=3, metric='cosine', leaf_size=30, radius=10, algorithm='brute').fit(tf_idf_matrix)    
#   with gzip.open(config['ngrams_vectorizerPath'], 'wb') as handle:
#       pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#   with gzip.open(config['ngrams_nbrsPath'], 'wb') as handle:
#       pickle.dump(nbrs, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  return nbrs,vectorizer

def fitNearestTFIDF(data, name):
  global config
  vectorizer = TfidfVectorizer(min_df=0, analyzer='word', lowercase=False)
  tf_idf_matrix = vectorizer.fit_transform(data)
  nbrs = NearestNeighbors(n_neighbors=20, metric='cosine', leaf_size=30, radius=10, algorithm='brute').fit(tf_idf_matrix)    
  with gzip.open(config['tfidf_vectorizerPath'], 'wb') as handle:
      pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  with gzip.open(config['tfidf_nbrsPath'], 'wb') as handle:
      pickle.dump(nbrs, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  return nbrs,vectorizer

def fitNearestTFIDFJaccard(data, name):
  global config
  vectorizer = CountVectorizer(min_df=1, analyzer='word', lowercase=False)
  tf_idf_matrix = vectorizer.fit_transform(data)
  nbrs = NearestNeighbors(n_neighbors=10, metric='jaccard', algorithm='brute').fit(tf_idf_matrix)    
  with gzip.open(config['tfidf_vectorizerPath'], 'wb') as handle:
      pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  with gzip.open(config['tfidf_nbrsPath'], 'wb') as handle:
      pickle.dump(nbrs, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  return nbrs,vectorizer

def getNearestN(query,vectorizer,nbrs):
    try:  
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_, n_neighbors=3)
    except Exception as e:
        queryTFIDF_ = queryTFIDF_.reshape(1, vectorizer.vocabulary_)
        distances, indices = nbrs.kneighbors(queryTFIDF_, n_neighbors=3)
    return distances, indices