import logging
import warnings
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import coo_array, csr_array, csr_matrix, hstack
from scipy.special import softmax
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

def least_squares_fit(features, target, scaling=1):
    if False:
        i = 10
        return i + 15
    X = features
    y = target.reshape(-1)
    zX = X.toarray()
    summed_target = y
    vote_matrix = csr_matrix(zX[:, :-1])
    constraint = LinearConstraint(np.ones(X.shape[-1] - 1), 1 * scaling, 1 * scaling)
    init = np.ones(X.shape[-1] - 1)
    init = init / np.linalg.norm(init)
    result = minimize(lambda x: np.sum((vote_matrix @ x - summed_target) ** 2), init, jac=lambda x: 2 * vote_matrix.T @ (vote_matrix @ x - summed_target), constraints=constraint, hess=lambda _: 2 * vote_matrix.T @ vote_matrix, method='trust-constr')
    return np.concatenate([result.x, np.ones(1)])

def get_df(study_label):
    if False:
        i = 10
        return i + 15
    conn = psycopg2.connect('host=0.0.0.0 port=5432 user=postgres password=postgres dbname=postgres')
    query = 'SELECT DISTINCT message_id, labels, message.user_id FROM text_labels JOIN message ON message_id = message.id;'
    df = pd.read_sql(query, con=conn)
    print(df.head())
    conn.close()
    users = set()
    messages = set()
    for row in df.itertuples(index=False):
        row = row._asdict()
        users.add(str(row['user_id']))
        messages.add(str(row['message_id']))
    users = list(users)
    messages = list(messages)
    print('num users:', len(users), 'num messages:', len(messages), 'num in df', len(df))
    row_idx = []
    col_idx = []
    data = []

    def swap(x):
        if False:
            while True:
                i = 10
        return (x[1], x[0])
    dct = dict(map(swap, enumerate(messages)))
    print('converting messages...')
    df['message_id'] = df['message_id'].map(dct)
    print('converting users...')
    df['user_id'] = df['user_id'].map(dict(map(swap, enumerate(users))))
    print('converting labels...')
    df['labels'] = df['labels'].map(lambda x: float(x.get(study_label, 0)))
    row_idx = df['message_id'].to_numpy()
    col_idx = df['user_id'].to_numpy()
    data = df['labels'].to_numpy()
    print(data)
    print(row_idx)
    print(col_idx)
    ' for row in df.itertuples(index=False):\n        row = row._asdict()\n        labels = row["labels"]\n        value = labels.get(study_label, None)\n        if value is not None:\n            # tmp=out[str(row["message_id"])]\n            # tmp = np.array(tmp)\n            # tmp[users.index(row["user_id"])] = value\n            # out[str(row["message_id"])] = np.array(tmp)\n            # print(out[str(row["message_id"])].density)\n            row_idx.append(messages.index(str(row["message_id"])))\n            col_idx.append(users.index(str(row["user_id"])))\n            data.append(value)\n            #arr[mid, uid] = value '
    arr = csr_array(coo_array((data, (row_idx, col_idx))))
    print('results', len(users), arr.shape)
    print('generated dataframe')
    return (arr, messages, users)

def reweight_features(features, weights, noise_scale=0.0):
    if False:
        while True:
            i = 10
    noise = np.random.randn(weights.shape[0]) * noise_scale
    weights = weights + noise
    values = features @ weights
    return values

def get_subframe(arr, columns_to_filter):
    if False:
        i = 10
        return i + 15
    '\n    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.\n    '
    if not isinstance(arr, csr_array):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    indices = list(columns_to_filter)
    mask = np.ones(arr.shape[1], dtype=bool)
    mask[indices] = False
    return arr[:, mask]

def sample_importance_weights(importance_weights, temperature=1.0):
    if False:
        i = 10
        return i + 15
    weights = softmax(abs(importance_weights) / temperature)
    column = np.random.choice(len(importance_weights), p=weights)
    return column

def make_random_testframe(num_rows, num_cols, frac_missing):
    if False:
        return 10
    data = np.random.rand(num_rows, num_cols).astype(np.float16)
    mask = np.random.rand(num_rows, num_cols) < frac_missing
    data[mask] = np.nan
    return data

def combine_underrepresented_columns(arr, num_instances):
    if False:
        return 10
    mask = arr != 0
    to_combine = mask.sum(0) < num_instances
    if not any(to_combine):
        return arr
    mean = np.mean(arr[:, to_combine], 1).reshape(-1, 1)
    dp = np.arange(len(to_combine))[to_combine]
    arr = get_subframe(arr, dp)
    arr = hstack([arr, mean])
    return arr

def importance_votes(arr, to_fit=10, init_weight=None):
    if False:
        while True:
            i = 10
    filtered_columns = []
    weighter = None
    if init_weight is None:
        weighter = np.ones(arr.shape[1]) / arr.shape[1]
    else:
        weighter = init_weight
    index = np.arange(arr.shape[1])
    bar = trange(to_fit)
    target = np.ones(arr.shape[0])
    for i in bar:
        index = list(filter(lambda x: x not in filtered_columns, index))
        target_old = target
        target = reweight_features(arr, weighter)
        error = np.mean((target - target_old) ** 2)
        bar.set_description(f'expected error: {error}', refresh=True)
        if error < 1e-10:
            break
        weighter = least_squares_fit(arr, target)
    return (reweight_features(arr, weighter), weighter)

def select_ids(arr, pick_frac, minima=(50, 500), folds=50, to_fit=200, frac=0.6):
    if False:
        print('Hello World!')
    '\n    selects the top-"pick_frac"% of messages from "arr" after merging all\n    users with less than "minima" votes (minima increases linearly with each iteration from min to max).\n    The method returns all messages that are within `frac` many "minima" selection\n    '
    votes = []
    minima = np.linspace(*minima, num=folds, dtype=int)
    num_per_iter = int(arr.shape[0] * pick_frac)
    writer_num = 0
    tmp = None
    for i in trange(folds):
        tofit = combine_underrepresented_columns(arr, minima[i])
        if tofit.shape[1] == writer_num:
            print('already tested these writer counts, skipping and using cached value.....')
            votes.append(tmp)
            continue
        writer_num = tofit.shape[1]
        init_weight = np.ones(tofit.shape[1]) / tofit.shape[1]
        (out, weight) = importance_votes(tofit, init_weight=init_weight, to_fit=to_fit)
        indices = np.argpartition(out, -num_per_iter)[-num_per_iter:]
        tmp = np.zeros(arr.shape[0])
        tmp[indices] = 1
        votes.append(tmp)
    out = []
    votes = np.stack(votes, axis=0)
    print('votespace', votes.shape)
    votes = np.mean(votes, 0)
    for (idx, f) in enumerate(votes):
        if f > frac:
            out.append((idx, f))
    return out
LOG = logging.getLogger(__name__)
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.simplefilter('ignore')
    logging.captureWarnings(True)
    logging.basicConfig(level=logging.ERROR)
    (df, message_ids, users) = get_df('quality')
    print('combining columns:')
    weights = np.ones(df.shape[-1])
    y = reweight_features(df, weights)
    num_per_iter = int(df.shape[0] * 0.5)
    naive = np.argpartition(y, -num_per_iter)[-num_per_iter:]
    print('after preprocessing')
    print('STARTING RUN')
    with logging_redirect_tqdm():
        print('selected ids')
        ids = select_ids(df, 0.5, folds=500)
    conn = psycopg2.connect('host=0.0.0.0 port=5432 user=postgres password=postgres dbname=postgres')
    query = 'SELECT DISTINCT id as message_id, message_tree_id FROM message;'
    print('selected', len(ids), 'messages')
    df = pd.read_sql(query, con=conn)
    out = []
    fracs = []
    in_naive = []
    for (i, frac) in ids:
        res = message_ids[i]
        out.append(df.loc[df['message_id'] == res])
        fracs.append(frac)
        in_naive.append(i in naive)
    df = pd.concat(out)
    df['fracs'] = fracs
    df['in_naive'] = in_naive
    print(df.shape)
    print('differences from naive', len(in_naive) - sum(in_naive))
    print(df)
    df.to_csv('output.csv')