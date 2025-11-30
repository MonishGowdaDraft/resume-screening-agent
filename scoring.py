import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def keyword_score(jd_text, resume_text, keywords=None):
    if keywords is None:
        keywords = list(set(re.findall(r"[A-Za-z+#\.\-]+", jd_text.lower())))

    resume_tokens = set(re.findall(r"[A-Za-z+#\.\-]+", resume_text.lower()))
    matches = sum(1 for k in keywords if k in resume_tokens)

    return matches / max(1, len(keywords))

def embedding_similarity(jd_emb, resume_emb):
    sim = cosine_similarity([jd_emb], [resume_emb])[0][0]
    return (sim + 1) / 2   # normalize [-1,1] → [0,1]

def experience_score(expected, actual):
    if expected <= 0:
        return min(1.0, actual / 10)
    if actual >= expected:
        return 1.0
    return actual / expected

def final_score(kw, emb, exp, weights=(0.45, 0.45, 0.10)):
    w1, w2, w3 = weights
    return w1 * kw + w2 * emb + w3 * exp
