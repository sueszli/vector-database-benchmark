import numpy as np
import pandas as pd
import fastai
import fastprogress
from fastprogress.fastprogress import force_console_behavior
from recommenders.utils import constants as cc

def cartesian_product(*arrays):
    if False:
        for i in range(10):
            print('nop')
    'Compute the Cartesian product in fastai algo. This is a helper function.\n\n    Args:\n        arrays (tuple of numpy.ndarray): Input arrays\n\n    Returns:\n        numpy.ndarray: product\n\n    '
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for (i, a) in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def score(learner, test_df, user_col=cc.DEFAULT_USER_COL, item_col=cc.DEFAULT_ITEM_COL, prediction_col=cc.DEFAULT_PREDICTION_COL, top_k=None):
    if False:
        i = 10
        return i + 15
    'Score all users+items provided and reduce to top_k items per user if top_k>0\n\n    Args:\n        learner (object): Model.\n        test_df (pandas.DataFrame): Test dataframe.\n        user_col (str): User column name.\n        item_col (str): Item column name.\n        prediction_col (str): Prediction column name.\n        top_k (int): Number of top items to recommend.\n\n    Returns:\n        pandas.DataFrame: Result of recommendation\n    '
    (total_users, total_items) = learner.data.train_ds.x.classes.values()
    test_df.loc[~test_df[user_col].isin(total_users), user_col] = np.nan
    test_df.loc[~test_df[item_col].isin(total_items), item_col] = np.nan
    u = learner.get_idx(test_df[user_col], is_item=False)
    m = learner.get_idx(test_df[item_col], is_item=True)
    pred = learner.model.forward(u, m)
    scores = pd.DataFrame({user_col: test_df[user_col], item_col: test_df[item_col], prediction_col: pred})
    scores = scores.sort_values([user_col, prediction_col], ascending=[True, False])
    if top_k is not None:
        top_scores = scores.groupby(user_col).head(top_k).reset_index(drop=True)
    else:
        top_scores = scores
    return top_scores

def hide_fastai_progress_bar():
    if False:
        return 10
    'Hide fastai progress bar'
    fastprogress.fastprogress.NO_BAR = True
    fastprogress.fastprogress.WRITER_FN = str
    (master_bar, progress_bar) = force_console_behavior()
    (fastai.basic_train.master_bar, fastai.basic_train.progress_bar) = (master_bar, progress_bar)