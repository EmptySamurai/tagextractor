from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tagextractor.native import get_tags_indices
from scipy.sparse import find

def get_tags(texts, n_tags, vectorizer=None, batchsize=8192):
    n_texts = len(texts)
    vectorizer = vectorizer or TfidfVectorizer(norm='l2')

    pred_tags = []
    for _ in range(n_texts):
        pred_tags.append([])

    tf_idf_matrix = vectorizer.fit_transform(texts)
    inverse_vocabulary=[None]*tf_idf_matrix.shape[1]
    for term, index in vectorizer.vocabulary_.items():
        inverse_vocabulary[index]=term
    inverse_vocabulary = np.array(inverse_vocabulary)

    rows, columns, values = find(tf_idf_matrix)
    tags_indices=get_tags_indices(rows, columns, values, tf_idf_matrix.shape[0], tf_idf_matrix.shape[1], n_tags, batchsize)

    for j, tags in enumerate(tags_indices):
        tags_batch=inverse_vocabulary[tags]
        for k in range(len(tags_batch)):
            pred_tags[j].append(tags_batch[k])

    return pred_tags