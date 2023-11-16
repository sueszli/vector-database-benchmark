"""
Visualizing cross-validation behavior in scikit-learn
=====================================================

Choosing the right cross-validation object is a crucial part of fitting a
model properly. There are many ways to split data into training and test
sets in order to avoid model overfitting, to standardize the number of
groups in test sets, etc.

This example visualizes the behavior of several common scikit-learn objects
for comparison.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4
n_points = 100
X = rng.randn(100, 10)
percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[ii] * int(100 * perc) for (ii, perc) in enumerate(percentiles_classes)])
group_prior = rng.dirichlet([2] * 10)
groups = np.repeat(np.arange(10), rng.multinomial(100, group_prior))

def visualize_groups(classes, groups, name):
    if False:
        print('Hello World!')
    (fig, ax) = plt.subplots()
    ax.scatter(range(len(groups)), [0.5] * len(groups), c=groups, marker='_', lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)), [3.5] * len(groups), c=classes, marker='_', lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[0.5, 3.5], yticklabels=['Data\ngroup', 'Data\nclass'], xlabel='Sample index')
visualize_groups(y, groups, 'no groups')

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    if False:
        return 10
    'Create a sample plot for indices of a cross-validation object.'
    for (ii, (tr, tt)) in enumerate(cv.split(X=X, y=y, groups=group)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv, vmin=-0.2, vmax=1.2)
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=cmap_data)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_', lw=lw, cmap=cmap_data)
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + 0.5, yticklabels=yticklabels, xlabel='Sample index', ylabel='CV iteration', ylim=[n_splits + 2.2, -0.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
(fig, ax) = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)
cvs = [StratifiedKFold, GroupKFold, StratifiedGroupKFold]
for cv in cvs:
    (fig, ax) = plt.subplots(figsize=(6, 3))
    plot_cv_indices(cv(n_splits), X, y, groups, ax, n_splits)
    ax.legend([Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))], ['Testing set', 'Training set'], loc=(1.02, 0.8))
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
cvs = [KFold, GroupKFold, ShuffleSplit, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit]
for cv in cvs:
    this_cv = cv(n_splits=n_splits)
    (fig, ax) = plt.subplots(figsize=(6, 3))
    plot_cv_indices(this_cv, X, y, groups, ax, n_splits)
    ax.legend([Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))], ['Testing set', 'Training set'], loc=(1.02, 0.8))
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
plt.show()