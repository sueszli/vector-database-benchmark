""" A script for running and testing nnmf
notes:
Total 1953447 tweets
"""
import json
import pandas as pd
from Util.Import import load_file, get_files
from model.nnmf import build_sparse_matrix, factorise, evaluate
from model import BM25
import time
import numpy as np

number_of_files = 125
number_of_topics = 10
iterations = 20
max_tweets = 1000
matrix_density = 0.1
convergence = 20
search_terms = "rugby world cup"



if __name__ == "__main__":


    search_term_lengths = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    matrix_densities = [0.25, 0.2, 0.15, 0.1, 0.05, 0.025]
    dfGrid = pd.DataFrame(np.zeros([len(search_term_lengths) * len(matrix_densities), 4]),columns=["NumTweets", "Density", "AvgTopicLen", "Time"])

    count = 0
    for max_tweets in search_term_lengths:
        for matrix_density in matrix_densities:
            tic = time.time()


            paths = get_files('./data/')

            # Comment out following line to run factorisation on entire dataset
            paths = paths[:number_of_files]

            length = len(paths)

            with open('./dictionaries/id_to_term_dictionary.json', 'r') as f:
                dict = json.load(f)

            unique_terms = len(dict.keys())

            matrix = []
            data = pd.DataFrame()

            keywords = search_terms.split()
            l = len(keywords)
            for i, path in enumerate(paths):
                print("Searching data: {:0.2%}".format(i / length), end='\r')
                data_temp = load_file(path)
                arr = "[" + ("keywords[%i].lower() in string.lower() or " * (l-1)) + "keywords[%i].lower() in string.lower()" + \
                  " for string in data_temp['text']]"
                arr = eval(arr % tuple([i for i in range(l)]))
                data_temp = data_temp[arr]
                data = pd.concat([data, data_temp])
            print("Data search complete.              ")
            print("{} tweets found for '{}'.\n".format(len(data), search_terms))

            print("Running BM25 to rank data.")
            data, matrix = BM25.BM25(data, keywords, 1.5, 0.5, max_tweets)
            print("Complete. {} tweets returned".format(len(data)))

            dfGrid.NumTweets.iloc[count] = len(matrix)
            print(len(matrix))
            dfGrid.Density.iloc[count] = matrix_density
            print(matrix_density)

            try:
                matrix = build_sparse_matrix(matrix, unique_terms, verbose=True)

                w, h = factorise(matrix, topics=number_of_topics, iterations=iterations, init_density=matrix_density,
                                 convergence=convergence)

                t = evaluate(w, dict)

                print("!!!!!!!!")
                dfGrid.AvgTopicLen.iloc[count] = (sum([len(i) for i in t])/len(t))

            except:
                pass

            toc = time.time()
            dfGrid.Time.iloc[count] = toc - tic
            print(dfGrid)
            dfGrid.to_csv("gridResults2.csv")
            count+=1
