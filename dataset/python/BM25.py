import math
import json
import pandas as pd
import numpy as np
from Util.adhoc_vectoriser import vectorise


def DocLength(Doc):
    """
    Calculates length of a document
    """
    return sum(Doc.values())


def AvgDocLength(Docs):
    """
   Calculates average length of all documents within a set
    """
    count = 0
    total = 0

    for Doc in Docs:
        total += DocLength(Doc)
        count += 1

    try:
        avg = total / count
    except:
        raise ZeroDivisionError("Query not in corpus")

    return avg


def MakeIDF(query, docs):
    """
    Cacuates the IDF portion of the code in which the inverse distribution function is calculated for each query
    """
    IDF = {}
    N = len(docs.keys())
    for term in query.keys():
        if term not in IDF:
            n = 0
            for key in docs:
                if str(term) in docs[key].keys():
                    n += 1
            idf = math.log((N - n + 0.5)/(n + 0.5), 2)
            IDF[term] = idf
    return IDF


def termFreq(term, doc):
    """
    Checks for a given term within the document and if it is present returns its frequency
    """
    if str(term) in doc.keys():
        return doc[str(term)]
    else:
        return 0.0


def calcBM25(query, doc, IDF, k, b, avgdl):
    """
    Iterates through the keys of the query scoring each individually before returning the sum of these scores
    """
    score = 0.0

    for key in query.keys():
        numer = termFreq(str(key), doc) * (k + 1.0)
        denom = termFreq(str(key), doc) + (k * (1.0 - b) + (b * DocLength(doc) / avgdl))
        score += IDF[str(key)] * (numer / denom)

    return score

def NDCG(df, K):
    rels = df["Score"]
    ideal_rels = np.sort(rels)[::-1]
    dcg = rels[0]
    for i in range(1, K):
        try:
            dcg += rels[i] / math.log(i+1, 2)
        except:
            pass

    idcg = ideal_rels[0]
    for i in range(1, K):
        try:
            idcg += ideal_rels[i] / math.log(i+1, 2)
        except:
            pass
    ndcg = dcg/idcg

    print("Search Results in an NDCG accuracy of: ", ndcg)

def BM25(data, keywords, k, b, max_tweets, eval = None, K = 1000):
    """
    Iterates through all docs calculating the BM25 scores for each query, saving these, having been
    ordered in the set file path.
    """
    matrix = []
    query_v = vectorise(keywords)
    with open('./dictionaries/idf_reference.json') as fp:
        IDF = json.load(fp)
    avgD = AvgDocLength(data.vector)
    data["BM25"] = data.vector.apply(lambda x: calcBM25(query_v, x, IDF, k, b, avgD))
    data = data.sort_values("BM25", ascending=False)
    data = data.reset_index()

    if eval != None:
        if eval == "rugby":
            df = pd.read_csv("./BM25_samples/rugby world cup_output.csv", usecols=[0,1])
            NDCG(df, K)
        elif eval == "fireworks":
            df = pd.read_csv("./BM25_samples/fireworks night_output.csv", usecols=[0,1])
            NDCG(df, K)
        else:
            print("No prepared data for evaluation")

    try:
        matrix += data['vector'][0:max_tweets].tolist()
        data = data.reset_index()
        data = data.ix[:max_tweets-1, :]
        return data, matrix
    except:
        matrix += data['vector'].tolist()
        return data, matrix
