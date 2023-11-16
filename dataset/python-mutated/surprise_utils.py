import pandas as pd
import numpy as np
from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL
from recommenders.utils.general_utils import invert_dictionary

def surprise_trainset_to_df(trainset, col_user='uid', col_item='iid', col_rating='rating'):
    if False:
        return 10
    'Converts a `surprise.Trainset` object to `pandas.DataFrame`\n\n    More info: https://surprise.readthedocs.io/en/stable/trainset.html\n\n    Args:\n        trainset (object): A surprise.Trainset object.\n        col_user (str): User column name.\n        col_item (str): Item column name.\n        col_rating (str): Rating column name.\n\n    Returns:\n        pandas.DataFrame: A dataframe with user column (str), item column (str), and rating column (float).\n    '
    df = pd.DataFrame(trainset.all_ratings(), columns=[col_user, col_item, col_rating])
    map_user = trainset._inner2raw_id_users if trainset._inner2raw_id_users is not None else invert_dictionary(trainset._raw2inner_id_users)
    map_item = trainset._inner2raw_id_items if trainset._inner2raw_id_items is not None else invert_dictionary(trainset._raw2inner_id_items)
    df[col_user] = df[col_user].map(map_user)
    df[col_item] = df[col_item].map(map_item)
    return df

def predict(algo, data, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL, predcol=DEFAULT_PREDICTION_COL):
    if False:
        print('Hello World!')
    'Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.\n\n    Args:\n        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise\n        data (pandas.DataFrame): the data on which to predict\n        usercol (str): name of the user column\n        itemcol (str): name of the item column\n\n    Returns:\n        pandas.DataFrame: Dataframe with usercol, itemcol, predcol\n    '
    predictions = [algo.predict(getattr(row, usercol), getattr(row, itemcol)) for row in data.itertuples()]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(index=str, columns={'uid': usercol, 'iid': itemcol, 'est': predcol})
    return predictions.drop(['details', 'r_ui'], axis='columns')

def compute_ranking_predictions(algo, data, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL, predcol=DEFAULT_PREDICTION_COL, remove_seen=False):
    if False:
        print('Hello World!')
    'Computes predictions of an algorithm from Surprise on all users and items in data. It can be used for computing\n    ranking metrics like NDCG.\n\n    Args:\n        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise\n        data (pandas.DataFrame): the data from which to get the users and items\n        usercol (str): name of the user column\n        itemcol (str): name of the item column\n        remove_seen (bool): flag to remove (user, item) pairs seen in the training data\n\n    Returns:\n        pandas.DataFrame: Dataframe with usercol, itemcol, predcol\n    '
    preds_lst = []
    users = data[usercol].unique()
    items = data[itemcol].unique()
    for user in users:
        for item in items:
            preds_lst.append([user, item, algo.predict(user, item).est])
    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])
    if remove_seen:
        tempdf = pd.concat([data[[usercol, itemcol]], pd.DataFrame(data=np.ones(data.shape[0]), columns=['dummycol'], index=data.index)], axis=1)
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how='outer')
        return merged[merged['dummycol'].isnull()].drop('dummycol', axis=1)
    else:
        return all_predictions