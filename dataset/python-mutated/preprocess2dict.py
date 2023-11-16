from __future__ import print_function, division
from builtins import range, input
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
df = pd.read_csv('../large_files/movielens-20m-dataset/very_small_rating.csv')
N = df.userId.max() + 1
M = df.movie_idx.max() + 1
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]
user2movie = {}
movie2user = {}
usermovie2rating = {}
print('Calling: update_user2movie_and_movie2user')
count = 0

def update_user2movie_and_movie2user(row):
    if False:
        while True:
            i = 10
    global count
    count += 1
    if count % 100000 == 0:
        print('processed: %.3f' % (float(count) / cutoff))
    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)
    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)
    usermovie2rating[i, j] = row.rating
df_train.apply(update_user2movie_and_movie2user, axis=1)
usermovie2rating_test = {}
print('Calling: update_usermovie2rating_test')
count = 0

def update_usermovie2rating_test(row):
    if False:
        print('Hello World!')
    global count
    count += 1
    if count % 100000 == 0:
        print('processed: %.3f' % (float(count) / len(df_test)))
    i = int(row.userId)
    j = int(row.movie_idx)
    usermovie2rating_test[i, j] = row.rating
df_test.apply(update_usermovie2rating_test, axis=1)
with open('user2movie.json', 'wb') as f:
    pickle.dump(user2movie, f)
with open('movie2user.json', 'wb') as f:
    pickle.dump(movie2user, f)
with open('usermovie2rating.json', 'wb') as f:
    pickle.dump(usermovie2rating, f)
with open('usermovie2rating_test.json', 'wb') as f:
    pickle.dump(usermovie2rating_test, f)