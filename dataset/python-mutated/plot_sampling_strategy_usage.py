"""
====================================================
How to use ``sampling_strategy`` in imbalanced-learn
====================================================

This example shows the different usage of the parameter ``sampling_strategy``
for the different family of samplers (i.e. over-sampling, under-sampling. or
cleaning methods).

"""
print(__doc__)
import seaborn as sns
sns.set_context('poster')
from sklearn.datasets import load_iris
from imblearn.datasets import make_imbalance
iris = load_iris(as_frame=True)
sampling_strategy = {0: 10, 1: 20, 2: 47}
(X, y) = make_imbalance(iris.data, iris.target, sampling_strategy=sampling_strategy)
import matplotlib.pyplot as plt
(fig, axs) = plt.subplots(ncols=2, figsize=(10, 5))
autopct = '%.2f'
iris.target.value_counts().plot.pie(autopct=autopct, ax=axs[0])
axs[0].set_title('Original')
y.value_counts().plot.pie(autopct=autopct, ax=axs[1])
axs[1].set_title('Imbalanced')
fig.tight_layout()
binary_mask = y.isin([0, 1])
binary_y = y[binary_mask]
binary_X = X[binary_mask]
from imblearn.under_sampling import RandomUnderSampler
sampling_strategy = 0.8
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = rus.fit_resample(binary_X, binary_y)
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title('Under-sampling')
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = ros.fit_resample(binary_X, binary_y)
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title('Over-sampling')
sampling_strategy = 'not minority'
(fig, axs) = plt.subplots(ncols=2, figsize=(10, 5))
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = rus.fit_resample(X, y)
y_res.value_counts().plot.pie(autopct=autopct, ax=axs[0])
axs[0].set_title('Under-sampling')
sampling_strategy = 'not majority'
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = ros.fit_resample(X, y)
y_res.value_counts().plot.pie(autopct=autopct, ax=axs[1])
_ = axs[1].set_title('Over-sampling')
from imblearn.under_sampling import TomekLinks
sampling_strategy = 'not minority'
tl = TomekLinks(sampling_strategy=sampling_strategy)
(X_res, y_res) = tl.fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title('Cleaning')
(fig, axs) = plt.subplots(ncols=2, figsize=(10, 5))
sampling_strategy = {0: 10, 1: 15, 2: 20}
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = rus.fit_resample(X, y)
y_res.value_counts().plot.pie(autopct=autopct, ax=axs[0])
axs[0].set_title('Under-sampling')
sampling_strategy = {0: 25, 1: 35, 2: 47}
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
(X_res, y_res) = ros.fit_resample(X, y)
y_res.value_counts().plot.pie(autopct=autopct, ax=axs[1])
_ = axs[1].set_title('Under-sampling')
sampling_strategy = [0, 1, 2]
tl = TomekLinks(sampling_strategy=sampling_strategy)
(X_res, y_res) = tl.fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title('Cleaning')

def ratio_multiplier(y):
    if False:
        print('Hello World!')
    from collections import Counter
    multiplier = {1: 0.7, 2: 0.95}
    target_stats = Counter(y)
    for (key, value) in target_stats.items():
        if key in multiplier:
            target_stats[key] = int(value * multiplier[key])
    return target_stats
(X_res, y_res) = RandomUnderSampler(sampling_strategy=ratio_multiplier).fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct=autopct)
ax.set_title('Under-sampling')
plt.show()