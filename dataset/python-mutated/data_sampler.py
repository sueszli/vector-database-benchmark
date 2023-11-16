"""Functions to create bandit problems from datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf

def one_hot(df, cols):
    if False:
        for i in range(10):
            print('nop')
    'Returns one-hot encoding of DataFrame df including columns in cols.'
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df

def sample_mushroom_data(file_name, num_contexts, r_noeat=0, r_eat_safe=5, r_eat_poison_bad=-35, r_eat_poison_good=5, prob_poison_bad=0.5):
    if False:
        print('Hello World!')
    'Samples bandit game from Mushroom UCI Dataset.\n\n  Args:\n    file_name: Route of file containing the original Mushroom UCI dataset.\n    num_contexts: Number of points to sample, i.e. (context, action rewards).\n    r_noeat: Reward for not eating a mushroom.\n    r_eat_safe: Reward for eating a non-poisonous mushroom.\n    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.\n    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.\n    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.\n\n  Returns:\n    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).\n    opt_vals: Vector of expected optimal (reward, action) for each context.\n\n  We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.\n  '
    df = pd.read_csv(file_name, header=None)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)
    contexts = df.iloc[ind, 2:]
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice([r_eat_poison_bad, r_eat_poison_good], p=[prob_poison_bad, 1 - prob_poison_bad], size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.reshape((num_contexts, 1))
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]
    if r_noeat > exp_eat_poison_reward:
        opt_actions = df.iloc[ind, 0]
    else:
        opt_actions = np.ones((num_contexts, 1))
    opt_vals = (opt_exp_reward.values, opt_actions.values)
    return (np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals)

def sample_stock_data(file_name, context_dim, num_actions, num_contexts, sigma, shuffle_rows=True):
    if False:
        i = 10
        return i + 15
    'Samples linear bandit game from stock prices dataset.\n\n  Args:\n    file_name: Route of file containing the stock prices dataset.\n    context_dim: Context dimension (i.e. vector with the price of each stock).\n    num_actions: Number of actions (different linear portfolio strategies).\n    num_contexts: Number of contexts to sample.\n    sigma: Vector with additive noise levels for each action.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).\n    opt_vals: Vector of expected optimal (reward, action) for each context.\n  '
    with tf.gfile.Open(file_name, 'r') as f:
        contexts = np.loadtxt(f, skiprows=1)
    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]
    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)
    mean_rewards = np.dot(contexts, betas)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise
    opt_actions = np.argmax(mean_rewards, axis=1)
    opt_rewards = [mean_rewards[i, a] for (i, a) in enumerate(opt_actions)]
    return (np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions))

def sample_jester_data(file_name, context_dim, num_actions, num_contexts, shuffle_rows=True, shuffle_cols=False):
    if False:
        print('Hello World!')
    'Samples bandit game from (user, joke) dense subset of Jester dataset.\n\n  Args:\n    file_name: Route of file containing the modified Jester dataset.\n    context_dim: Context dimension (i.e. vector with some ratings from a user).\n    num_actions: Number of actions (number of joke ratings to predict).\n    num_contexts: Number of contexts to sample.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n    shuffle_cols: Whether or not context/action jokes are randomly shuffled.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).\n    opt_vals: Vector of deterministic optimal (reward, action) for each context.\n  '
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for (i, a) in enumerate(opt_actions)])
    return (dataset, (opt_rewards, opt_actions))

def sample_statlog_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
    if False:
        return 10
    'Returns bandit problem dataset based on the UCI statlog data.\n\n  Args:\n    file_name: Route of file containing the Statlog dataset.\n    num_contexts: Number of contexts to sample.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n    remove_underrepresented: If True, removes arms with very few rewards.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, action rewards).\n    opt_vals: Vector of deterministic optimal (reward, action) for each context.\n\n  https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)\n  '
    with tf.gfile.Open(file_name, 'r') as f:
        data = np.loadtxt(f)
    num_actions = 7
    if shuffle_rows:
        np.random.shuffle(data)
    data = data[:num_contexts, :]
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1
    if remove_underrepresented:
        (contexts, labels) = remove_underrepresented_classes(contexts, labels)
    return classification_to_bandit_problem(contexts, labels, num_actions)

def sample_adult_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
    if False:
        print('Hello World!')
    'Returns bandit problem dataset based on the UCI adult data.\n\n  Args:\n    file_name: Route of file containing the Adult dataset.\n    num_contexts: Number of contexts to sample.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n    remove_underrepresented: If True, removes arms with very few rewards.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, action rewards).\n    opt_vals: Vector of deterministic optimal (reward, action) for each context.\n\n  Preprocessing:\n    * drop rows with missing values\n    * convert categorical variables to 1 hot encoding\n\n  https://archive.ics.uci.edu/ml/datasets/census+income\n  '
    with tf.gfile.Open(file_name, 'r') as f:
        df = pd.read_csv(f, header=None, na_values=[' ?']).dropna()
    num_actions = 14
    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]
    labels = df[6].astype('category').cat.codes.as_matrix()
    df = df.drop([6], axis=1)
    cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
    df = pd.get_dummies(df, columns=cols_to_transform)
    if remove_underrepresented:
        (df, labels) = remove_underrepresented_classes(df, labels)
    contexts = df.as_matrix()
    return classification_to_bandit_problem(contexts, labels, num_actions)

def sample_census_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
    if False:
        return 10
    "Returns bandit problem dataset based on the UCI census data.\n\n  Args:\n    file_name: Route of file containing the Census dataset.\n    num_contexts: Number of contexts to sample.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n    remove_underrepresented: If True, removes arms with very few rewards.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, action rewards).\n    opt_vals: Vector of deterministic optimal (reward, action) for each context.\n\n  Preprocessing:\n    * drop rows with missing labels\n    * convert categorical variables to 1 hot encoding\n\n  Note: this is the processed (not the 'raw') dataset. It contains a subset\n  of the raw features and they've all been discretized.\n\n  https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29\n  "
    with tf.gfile.Open(file_name, 'r') as f:
        df = pd.read_csv(f, header=0, na_values=['?']).dropna()
    num_actions = 9
    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]
    labels = df['dOccup'].astype('category').cat.codes.as_matrix()
    df = df.drop(['dOccup', 'caseid'], axis=1)
    df = pd.get_dummies(df, columns=df.columns)
    if remove_underrepresented:
        (df, labels) = remove_underrepresented_classes(df, labels)
    contexts = df.as_matrix()
    return classification_to_bandit_problem(contexts, labels, num_actions)

def sample_covertype_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns bandit problem dataset based on the UCI Cover_Type data.\n\n  Args:\n    file_name: Route of file containing the Covertype dataset.\n    num_contexts: Number of contexts to sample.\n    shuffle_rows: If True, rows from original dataset are shuffled.\n    remove_underrepresented: If True, removes arms with very few rewards.\n\n  Returns:\n    dataset: Sampled matrix with rows: (context, action rewards).\n    opt_vals: Vector of deterministic optimal (reward, action) for each context.\n\n  Preprocessing:\n    * drop rows with missing labels\n    * convert categorical variables to 1 hot encoding\n\n  https://archive.ics.uci.edu/ml/datasets/Covertype\n  '
    with tf.gfile.Open(file_name, 'r') as f:
        df = pd.read_csv(f, header=0, na_values=['?']).dropna()
    num_actions = 7
    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]
    labels = df[df.columns[-1]].astype('category').cat.codes.as_matrix()
    df = df.drop([df.columns[-1]], axis=1)
    if remove_underrepresented:
        (df, labels) = remove_underrepresented_classes(df, labels)
    contexts = df.as_matrix()
    return classification_to_bandit_problem(contexts, labels, num_actions)

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    if False:
        print('Hello World!')
    'Normalize contexts and encode deterministic rewards.'
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])
    contexts = (contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0
    return (contexts, rewards, (np.ones(num_contexts), labels))

def safe_std(values):
    if False:
        while True:
            i = 10
    'Remove zero std values for ones.'
    return np.array([val if val != 0.0 else 1.0 for val in values])

def remove_underrepresented_classes(features, labels, thresh=0.0005):
    if False:
        while True:
            i = 10
    'Removes classes when number of datapoints fraction is below a threshold.'
    total_count = labels.shape[0]
    (unique, counts) = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return (features[keep], labels[np.array(keep)])