import requests
from dagster import asset
import pandas as pd

@asset
def hackernews_stories():
    if False:
        return 10
    latest_item = requests.get('https://hacker-news.firebaseio.com/v0/maxitem.json').json()
    results = []
    scope = range(latest_item - 1000, latest_item)
    for item_id in scope:
        item = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json').json()
        results.append(item)
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df[df.type == 'story']
        df = df[~df.title.isna()]
    return df
from sklearn.model_selection import train_test_split
from dagster import multi_asset, AssetOut

@multi_asset(outs={'training_data': AssetOut(), 'test_data': AssetOut()})
def training_test_data(hackernews_stories):
    if False:
        print('Hello World!')
    X = hackernews_stories.title
    y = hackernews_stories.descendants
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    return ((X_train, y_train), (X_test, y_test))
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

@multi_asset(outs={'tfidf_vectorizer': AssetOut(), 'transformed_training_data': AssetOut()})
def transformed_train_data(training_data):
    if False:
        for i in range(10):
            print('nop')
    (X_train, y_train) = training_data
    vectorizer = TfidfVectorizer()
    transformed_X_train = vectorizer.fit_transform(X_train)
    transformed_X_train = transformed_X_train.toarray()
    y_train = y_train.fillna(0)
    transformed_y_train = np.array(y_train)
    return (vectorizer, (transformed_X_train, transformed_y_train))

@asset
def transformed_test_data(test_data, tfidf_vectorizer):
    if False:
        for i in range(10):
            print('nop')
    (X_test, y_test) = test_data
    transformed_X_test = tfidf_vectorizer.transform(X_test)
    transformed_y_test = np.array(y_test)
    y_test = y_test.fillna(0)
    transformed_y_test = np.array(y_test)
    return (transformed_X_test, transformed_y_test)
import xgboost as xg
from sklearn.metrics import mean_absolute_error

@asset
def xgboost_comments_model(transformed_training_data):
    if False:
        while True:
            i = 10
    (transformed_X_train, transformed_y_train) = transformed_training_data
    xgb_r = xg.XGBRegressor(objective='reg:squarederror', eval_metric=mean_absolute_error, n_estimators=20)
    xgb_r.fit(transformed_X_train, transformed_y_train)
    return xgb_r

@asset
def comments_model_test_set_r_squared(transformed_test_data, xgboost_comments_model):
    if False:
        return 10
    (transformed_X_test, transformed_y_test) = transformed_test_data
    score = xgboost_comments_model.score(transformed_X_test, transformed_y_test)
    return score

@asset
def latest_story_comment_predictions(xgboost_comments_model, tfidf_vectorizer):
    if False:
        return 10
    latest_item = requests.get('https://hacker-news.firebaseio.com/v0/maxitem.json').json()
    results = []
    scope = range(latest_item - 100, latest_item)
    for item_id in scope:
        item = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json').json()
        results.append(item)
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df[df.type == 'story']
        df = df[~df.title.isna()]
    inference_x = df.title
    inference_x = tfidf_vectorizer.transform(inference_x)
    return xgboost_comments_model.predict(inference_x)