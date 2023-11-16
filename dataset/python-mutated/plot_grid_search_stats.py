"""
==================================================
Statistical comparison of models using grid search
==================================================

This example illustrates how to statistically compare the performance of models
trained and evaluated using :class:`~sklearn.model_selection.GridSearchCV`.

"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
(X, y) = make_moons(noise=0.352, random_state=1, n_samples=100)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, marker='o', s=25, edgecolor='k', legend=False).set_title('Data')
plt.show()
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
param_grid = [{'kernel': ['linear']}, {'kernel': ['poly'], 'degree': [2, 3]}, {'kernel': ['rbf']}]
svc = SVC(random_state=0)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='roc_auc', cv=cv)
search.fit(X, y)
import pandas as pd
results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values(by=['rank_test_score'])
results_df = results_df.set_index(results_df['params'].apply(lambda x: '_'.join((str(val) for val in x.values())))).rename_axis('kernel')
results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']]
model_scores = results_df.filter(regex='split\\d*_test_score')
(fig, ax) = plt.subplots()
sns.lineplot(data=model_scores.transpose().iloc[:30], dashes=False, palette='Set1', marker='o', alpha=0.5, ax=ax)
ax.set_xlabel('CV test fold', size=12, labelpad=10)
ax.set_ylabel('Model AUC', size=12)
ax.tick_params(bottom=True, labelbottom=False)
plt.show()
print(f'Correlation of models:\n {model_scores.transpose().corr()}')
import numpy as np
from scipy.stats import t

def corrected_std(differences, n_train, n_test):
    if False:
        while True:
            i = 10
    "Corrects standard deviation using Nadeau and Bengio's approach.\n\n    Parameters\n    ----------\n    differences : ndarray of shape (n_samples,)\n        Vector containing the differences in the score metrics of two models.\n    n_train : int\n        Number of samples in the training set.\n    n_test : int\n        Number of samples in the testing set.\n\n    Returns\n    -------\n    corrected_std : float\n        Variance-corrected standard deviation of the set of differences.\n    "
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, df, n_train, n_test):
    if False:
        for i in range(10):
            print('nop')
    'Computes right-tailed paired t-test with corrected variance.\n\n    Parameters\n    ----------\n    differences : array-like of shape (n_samples,)\n        Vector containing the differences in the score metrics of two models.\n    df : int\n        Degrees of freedom.\n    n_train : int\n        Number of samples in the training set.\n    n_test : int\n        Number of samples in the testing set.\n\n    Returns\n    -------\n    t_stat : float\n        Variance-corrected t-statistic.\n    p_val : float\n        Variance-corrected p-value.\n    '
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)
    return (t_stat, p_val)
model_1_scores = model_scores.iloc[0].values
model_2_scores = model_scores.iloc[1].values
differences = model_1_scores - model_2_scores
n = differences.shape[0]
df = n - 1
n_train = len(list(cv.split(X, y))[0][0])
n_test = len(list(cv.split(X, y))[0][1])
(t_stat, p_val) = compute_corrected_ttest(differences, df, n_train, n_test)
print(f'Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}')
t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)
print(f'Uncorrected t-value: {t_stat_uncorrected:.3f}\nUncorrected p-value: {p_val_uncorrected:.3f}')
t_post = t(df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test))
x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 100)
plt.plot(x, t_post.pdf(x))
plt.xticks(np.arange(-0.04, 0.06, 0.01))
plt.fill_between(x, t_post.pdf(x), 0, facecolor='blue', alpha=0.2)
plt.ylabel('Probability density')
plt.xlabel('Mean difference ($\\mu$)')
plt.title('Posterior distribution')
plt.show()
better_prob = 1 - t_post.cdf(0)
print(f'Probability of {model_scores.index[0]} being more accurate than {model_scores.index[1]}: {better_prob:.3f}')
print(f'Probability of {model_scores.index[1]} being more accurate than {model_scores.index[0]}: {1 - better_prob:.3f}')
rope_interval = [-0.01, 0.01]
rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
print(f'Probability of {model_scores.index[0]} and {model_scores.index[1]} being practically equivalent: {rope_prob:.3f}')
x_rope = np.linspace(rope_interval[0], rope_interval[1], 100)
plt.plot(x, t_post.pdf(x))
plt.xticks(np.arange(-0.04, 0.06, 0.01))
plt.vlines([-0.01, 0.01], ymin=0, ymax=np.max(t_post.pdf(x)) + 1)
plt.fill_between(x_rope, t_post.pdf(x_rope), 0, facecolor='blue', alpha=0.2)
plt.ylabel('Probability density')
plt.xlabel('Mean difference ($\\mu$)')
plt.title('Posterior distribution under the ROPE')
plt.show()
cred_intervals = []
intervals = [0.5, 0.75, 0.95]
for interval in intervals:
    cred_interval = list(t_post.interval(interval))
    cred_intervals.append([interval, cred_interval[0], cred_interval[1]])
cred_int_df = pd.DataFrame(cred_intervals, columns=['interval', 'lower value', 'upper value']).set_index('interval')
cred_int_df
from itertools import combinations
from math import factorial
n_comparisons = factorial(len(model_scores)) / (factorial(2) * factorial(len(model_scores) - 2))
pairwise_t_test = []
for (model_i, model_k) in combinations(range(len(model_scores)), 2):
    model_i_scores = model_scores.iloc[model_i].values
    model_k_scores = model_scores.iloc[model_k].values
    differences = model_i_scores - model_k_scores
    (t_stat, p_val) = compute_corrected_ttest(differences, df, n_train, n_test)
    p_val *= n_comparisons
    p_val = 1 if p_val > 1 else p_val
    pairwise_t_test.append([model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val])
pairwise_comp_df = pd.DataFrame(pairwise_t_test, columns=['model_1', 'model_2', 't_stat', 'p_val']).round(3)
pairwise_comp_df
pairwise_bayesian = []
for (model_i, model_k) in combinations(range(len(model_scores)), 2):
    model_i_scores = model_scores.iloc[model_i].values
    model_k_scores = model_scores.iloc[model_k].values
    differences = model_i_scores - model_k_scores
    t_post = t(df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test))
    worse_prob = t_post.cdf(rope_interval[0])
    better_prob = 1 - t_post.cdf(rope_interval[1])
    rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
    pairwise_bayesian.append([worse_prob, better_prob, rope_prob])
pairwise_bayesian_df = pd.DataFrame(pairwise_bayesian, columns=['worse_prob', 'better_prob', 'rope_prob']).round(3)
pairwise_comp_df = pairwise_comp_df.join(pairwise_bayesian_df)
pairwise_comp_df