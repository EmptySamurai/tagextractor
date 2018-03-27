import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from tagextractor import get_tags
import gc
import os


def f1(y, y_pred):
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(y.shape[0]):
        yi = y[i] # it is a list
        ypi = y_pred[i] # it is a list
                
        for tag in yi:
            if tag in ypi:
                tp += 1
            else:
                fn += 1
        
        for tag in ypi:
            if not tag in yi:
                fp += 1
                
                
    p = (tp*1.) / (tp+fp)
    r = (tp*1.) / (tp+fn)
    f1 = (2*p*r)/(p+r)
    return f1



N_TAGS=3

df =  pd.read_csv(os.path.join(os.path.dirname(__file__),"robotics.csv"))
ids = df["id"].as_matrix()
n_samples=len(df)
print("Number of samples:",n_samples)


texts = []
for i, row in df.iterrows():
    words = str(row.title) + ' ' + str(row.content)
    texts.append(words)
texts=np.array(texts)

true_tags = []
for index, row in df.iterrows():
    true_tags.append(row["tags"].split())    
true_tags=np.array(true_tags)


print("Start time: {0}".format(time.ctime()))
print()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.1)
pred_tags = get_tags(texts, N_TAGS, vectorizer=vectorizer)

print("Finish time: {0}".format(time.ctime()))
print()
gc.collect()


for text_tags in pred_tags:
    for i in range(len(text_tags)):
        text_tags[i] = text_tags[i].replace(' ', '-').lower()

print("F1:",f1(true_tags, pred_tags))
