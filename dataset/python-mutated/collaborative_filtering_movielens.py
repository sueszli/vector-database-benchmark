"""
Title: Collaborative Filtering for Movie Recommendations
Author: [Siddhartha Banerjee](https://twitter.com/sidd2006)
Date created: 2020/05/24
Last modified: 2020/05/24
Description: Recommending movies using a model trained on Movielens dataset.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates\n[Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)\nusing the [Movielens dataset](https://www.kaggle.com/c/movielens-100k)\nto recommend movies to users.\nThe MovieLens ratings dataset lists the ratings given by a set of users to a set of movies.\nOur goal is to be able to predict ratings for movies a user has not yet watched.\nThe movies with the highest predicted ratings can then be recommended to the user.\n\nThe steps in the model are as follows:\n\n1. Map user ID to a "user vector" via an embedding matrix\n2. Map movie ID to a "movie vector" via an embedding matrix\n3. Compute the dot product between the user vector and movie vector, to obtain\nthe a match score between the user and the movie (predicted rating).\n4. Train the embeddings via gradient descent using all known user-movie pairs.\n\n**References:**\n\n- [Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/371920.372071)\n- [Neural Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)\n'
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile
import keras
from keras import layers
from keras import ops
'\n## First, load the data and apply preprocessing\n'
movielens_data_file_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
movielens_zipped_file = keras.utils.get_file('ml-latest-small.zip', movielens_data_file_url, extract=False)
keras_datasets_path = Path(movielens_zipped_file).parents[0]
movielens_dir = keras_datasets_path / 'ml-latest-small'
if not movielens_dir.exists():
    with ZipFile(movielens_zipped_file, 'r') as zip:
        print('Extracting all the files now...')
        zip.extractall(path=keras_datasets_path)
        print('Done!')
ratings_file = movielens_dir / 'ratings.csv'
df = pd.read_csv(ratings_file)
'\nFirst, need to perform some preprocessing to encode users and movies as integer indices.\n'
user_ids = df['userId'].unique().tolist()
user2user_encoded = {x: i for (i, x) in enumerate(user_ids)}
userencoded2user = {i: x for (i, x) in enumerate(user_ids)}
movie_ids = df['movieId'].unique().tolist()
movie2movie_encoded = {x: i for (i, x) in enumerate(movie_ids)}
movie_encoded2movie = {i: x for (i, x) in enumerate(movie_ids)}
df['user'] = df['userId'].map(user2user_encoded)
df['movie'] = df['movieId'].map(movie2movie_encoded)
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df['rating'] = df['rating'].values.astype(np.float32)
min_rating = min(df['rating'])
max_rating = max(df['rating'])
print('Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}'.format(num_users, num_movies, min_rating, max_rating))
'\n## Prepare training and validation data\n'
df = df.sample(frac=1, random_state=42)
x = df[['user', 'movie']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.9 * df.shape[0])
(x_train, x_val, y_train, y_val) = (x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])
'\n## Create the model\n\nWe embed both users and movies in to 50-dimensional vectors.\n\nThe model computes a match score between user and movie embeddings via a dot product,\nand adds a per-movie and per-user bias. The match score is scaled to the `[0, 1]`\ninterval via a sigmoid (since our ratings are normalized to this range).\n'
EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-06))
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-06))
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        if False:
            return 10
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = ops.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return ops.nn.sigmoid(x)
model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
'\n## Train the model based on the data split\n'
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_val, y_val))
'\n## Plot training and validation loss\n'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'\n## Show top 10 movie recommendations to a user\n'
movie_df = pd.read_csv(movielens_dir / 'movies.csv')
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[~movie_df['movieId'].isin(movies_watched_by_user.movieId.values)]['movieId']
movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]
print('Showing recommendations for user: {}'.format(user_id))
print('====' * 9)
print('Movies with high ratings from user')
print('----' * 8)
top_movies_user = movies_watched_by_user.sort_values(by='rating', ascending=False).head(5).movieId.values
movie_df_rows = movie_df[movie_df['movieId'].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ':', row.genres)
print('----' * 8)
print('Top 10 movie recommendations')
print('----' * 8)
recommended_movies = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ':', row.genres)
'\n**Example available on HuggingFace**\n\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Collaborative%20Filtering-black.svg)](https://huggingface.co/keras-io/collaborative-filtering-movielens) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Collaborative%20Filtering-black.svg)](https://huggingface.co/spaces/keras-io/collaborative-filtering-movielens) |\n'