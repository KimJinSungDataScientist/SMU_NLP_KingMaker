import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def txt2simg(df, DESCR):
        
    df = df.append({'king' : 'Input Image', 'description' : DESCR}, ignore_index=True)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    txt2idx = {}
    for i, c in enumerate(df['description']):
        txt2idx[i] = c

    idx2txt = {}
    for i, c in txt2idx.items():
        idx2txt[c] = i
        
    idx = idx2txt[DESCR]
    sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]

    # text 유사도 높은 순서대로 index 뽑기
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    return (sim_scores[0][0])