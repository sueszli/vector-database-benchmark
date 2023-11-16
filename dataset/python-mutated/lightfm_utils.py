import pandas as pd
import numpy as np
import seaborn as sns
from lightfm.evaluation import precision_at_k, recall_at_k

def model_perf_plots(df):
    if False:
        print('Hello World!')
    "Function to plot model performance metrics.\n\n    Args:\n        df (pandas.DataFrame): Dataframe in tidy format, with ['epoch','level','value'] columns\n\n    Returns:\n        object: matplotlib axes\n    "
    g = sns.FacetGrid(df, col='metric', hue='stage', col_wrap=2, sharey=False)
    g = g.map(sns.scatterplot, 'epoch', 'value').add_legend()

def compare_metric(df_list, metric='prec', stage='test'):
    if False:
        i = 10
        return i + 15
    'Function to combine and prepare list of dataframes into tidy format.\n\n    Args:\n        df_list (list): List of dataframes\n        metrics (str): name of metric to be extracted, optional\n        stage (str): name of model fitting stage to be extracted, optional\n\n    Returns:\n        pandas.DataFrame: Metrics\n    '
    colnames = ['model' + str(x) for x in list(range(1, len(df_list) + 1))]
    models = [df[(df['stage'] == stage) & (df['metric'] == metric)]['value'].reset_index(drop=True).values for df in df_list]
    output = pd.DataFrame(zip(*models), columns=colnames).stack().reset_index()
    output.columns = ['epoch', 'data', 'value']
    return output

def track_model_metrics(model, train_interactions, test_interactions, k=10, no_epochs=100, no_threads=8, show_plot=True, **kwargs):
    if False:
        print('Hello World!')
    "Function to record model's performance at each epoch, formats the performance into tidy format,\n    plots the performance and outputs the performance data.\n\n    Args:\n        model (LightFM instance): fitted LightFM model\n        train_interactions (scipy sparse COO matrix): train interactions set\n        test_interactions (scipy sparse COO matrix): test interaction set\n        k (int): number of recommendations, optional\n        no_epochs (int): Number of epochs to run, optional\n        no_threads (int): Number of parallel threads to use, optional\n        **kwargs: other keyword arguments to be passed down\n\n    Returns:\n        pandas.DataFrame, LightFM model, matplotlib axes:\n        - Performance traces of the fitted model\n        - Fitted model\n        - Side effect of the method\n    "
    model_prec_train = [0] * no_epochs
    model_prec_test = [0] * no_epochs
    model_rec_train = [0] * no_epochs
    model_rec_test = [0] * no_epochs
    for epoch in range(no_epochs):
        model.fit_partial(interactions=train_interactions, epochs=1, num_threads=no_threads, **kwargs)
        model_prec_train[epoch] = precision_at_k(model, train_interactions, k=k, **kwargs).mean()
        model_prec_test[epoch] = precision_at_k(model, test_interactions, k=k, **kwargs).mean()
        model_rec_train[epoch] = recall_at_k(model, train_interactions, k=k, **kwargs).mean()
        model_rec_test[epoch] = recall_at_k(model, test_interactions, k=k, **kwargs).mean()
    fitting_metrics = pd.DataFrame(zip(model_prec_train, model_prec_test, model_rec_train, model_rec_test), columns=['model_prec_train', 'model_prec_test', 'model_rec_train', 'model_rec_test'])
    fitting_metrics = fitting_metrics.stack().reset_index()
    fitting_metrics.columns = ['epoch', 'level', 'value']
    fitting_metrics['stage'] = fitting_metrics.level.str.split('_').str[-1]
    fitting_metrics['metric'] = fitting_metrics.level.str.split('_').str[1]
    fitting_metrics.drop(['level'], axis=1, inplace=True)
    metric_keys = {'prec': 'Precision', 'rec': 'Recall'}
    fitting_metrics.metric.replace(metric_keys, inplace=True)
    if show_plot:
        model_perf_plots(fitting_metrics)
    return (fitting_metrics, model)

def similar_users(user_id, user_features, model, N=10):
    if False:
        i = 10
        return i + 15
    'Function to return top N similar users based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681\n\n     Args:\n        user_id (int): id of user to be used as reference\n        user_features (scipy sparse CSR matrix): user feature matric\n        model (LightFM instance): fitted LightFM model\n        N (int): Number of top similar users to return\n\n    Returns:\n        pandas.DataFrame: top N most similar users with score\n    '
    (_, user_representations) = model.get_user_representations(features=user_features)
    scores = user_representations.dot(user_representations[user_id, :])
    user_norms = np.linalg.norm(user_representations, axis=1)
    user_norms[user_norms == 0] = 1e-10
    scores /= user_norms
    best = np.argpartition(scores, -(N + 1))[-(N + 1):]
    return pd.DataFrame(sorted(zip(best, scores[best] / user_norms[user_id]), key=lambda x: -x[1])[1:], columns=['userID', 'score'])

def similar_items(item_id, item_features, model, N=10):
    if False:
        for i in range(10):
            print('nop')
    'Function to return top N similar items\n    based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681\n\n    Args:\n        item_id (int): id of item to be used as reference\n        item_features (scipy sparse CSR matrix): item feature matric\n        model (LightFM instance): fitted LightFM model\n        N (int): Number of top similar items to return\n\n    Returns:\n        pandas.DataFrame: top N most similar items with score\n    '
    (_, item_representations) = model.get_item_representations(features=item_features)
    scores = item_representations.dot(item_representations[item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms
    best = np.argpartition(scores, -(N + 1))[-(N + 1):]
    return pd.DataFrame(sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])[1:], columns=['itemID', 'score'])

def prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights):
    if False:
        for i in range(10):
            print('nop')
    'Function to prepare test df for evaluation\n\n    Args:\n        test_idx (slice): slice of test indices\n        uids (numpy.ndarray): Array of internal user indices\n        iids (numpy.ndarray): Array of internal item indices\n        uid_map (dict): Keys to map internal user indices to external ids.\n        iid_map (dict): Keys to map internal item indices to external ids.\n        weights (numpy.float32 coo_matrix): user-item interaction\n\n    Returns:\n        pandas.DataFrame: user-item selected for testing\n    '
    test_df = pd.DataFrame(zip(uids[test_idx], iids[test_idx], [list(uid_map.keys())[x] for x in uids[test_idx]], [list(iid_map.keys())[x] for x in iids[test_idx]]), columns=['uid', 'iid', 'userID', 'itemID'])
    dok_weights = weights.todok()
    test_df['rating'] = test_df.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)
    return test_df[['userID', 'itemID', 'rating']]

def prepare_all_predictions(data, uid_map, iid_map, interactions, model, num_threads, user_features=None, item_features=None):
    if False:
        while True:
            i = 10
    'Function to prepare all predictions for evaluation.\n    Args:\n        data (pandas df): dataframe of all users, items and ratings as loaded\n        uid_map (dict): Keys to map internal user indices to external ids.\n        iid_map (dict): Keys to map internal item indices to external ids.\n        interactions (np.float32 coo_matrix): user-item interaction\n        model (LightFM instance): fitted LightFM model\n        num_threads (int): number of parallel computation threads\n        user_features (np.float32 csr_matrix): User weights over features\n        item_features (np.float32 csr_matrix):  Item weights over features\n    Returns:\n        pandas.DataFrame: all predictions\n    '
    (users, items, preds) = ([], [], [])
    item = list(data.itemID.unique())
    for user in data.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
    all_predictions = pd.DataFrame(data={'userID': users, 'itemID': items})
    all_predictions['uid'] = all_predictions.userID.map(uid_map)
    all_predictions['iid'] = all_predictions.itemID.map(iid_map)
    dok_weights = interactions.todok()
    all_predictions['rating'] = all_predictions.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)
    all_predictions = all_predictions[all_predictions.rating < 1].reset_index(drop=True)
    all_predictions = all_predictions.drop('rating', axis=1)
    all_predictions['prediction'] = all_predictions.apply(lambda x: model.predict(user_ids=np.array([x['uid']], dtype=np.int32), item_ids=np.array([x['iid']], dtype=np.int32), user_features=user_features, item_features=item_features, num_threads=num_threads)[0], axis=1)
    return all_predictions[['userID', 'itemID', 'prediction']]