import pandas as pd
from recommenders.utils.constants import DEFAULT_K

def compute_test_results(model, train, test, rating_metrics, ranking_metrics, k=DEFAULT_K):
    if False:
        return 10
    'Compute the test results using a trained NCF model.\n\n    Args:\n        model (object): TF model.\n        train (pandas.DataFrame): Train set.\n        test (pandas.DataFrame): Test set.\n        rating_metrics (list): List of rating metrics.\n        ranking_metrics (list): List of ranking metrics.\n        k (int): top K recommendations\n\n    Returns:\n        dict: Test results.\n\n    '
    test_results = {}
    predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)] for (_, row) in test.iterrows()]
    predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
    predictions = predictions.astype({'userID': 'int64', 'itemID': 'int64', 'prediction': 'float64'})
    for metric in rating_metrics:
        test_results[metric] = eval(metric)(test, predictions)
    (users, items, preds) = ([], [], [])
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))
    all_predictions = pd.DataFrame(data={'userID': users, 'itemID': items, 'prediction': preds})
    merged = pd.merge(train, all_predictions, on=['userID', 'itemID'], how='outer')
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
    for metric in ranking_metrics:
        test_results[metric] = eval(metric)(test, all_predictions, col_prediction='prediction', k=k)
    return test_results

def combine_metrics_dicts(*metrics):
    if False:
        print('Hello World!')
    'Combine metrics from dicts.\n\n    Args:\n        metrics (dict): Metrics\n\n    Returns:\n        pandas.DataFrame: Dataframe with metrics combined.\n    '
    df = pd.DataFrame(metrics[0], index=[0])
    for metric in metrics[1:]:
        df = df.append(pd.DataFrame(metric, index=[0]), sort=False)
    return df