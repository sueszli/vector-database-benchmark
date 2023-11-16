"""
===============================
Compare under-sampling samplers
===============================

The following example attends to make a qualitative comparison between the
different under-sampling algorithms available in the imbalanced-learn package.
"""
print(__doc__)
import seaborn as sns
sns.set_context('poster')
from sklearn.datasets import make_classification

def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3, class_sep=0.8, n_clusters=1):
    if False:
        return 10
    return make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=n_classes, n_clusters_per_class=n_clusters, weights=list(weights), class_sep=class_sep, random_state=0)

def plot_resampling(X, y, sampler, ax, title=None):
    if False:
        for i in range(10):
            print('nop')
    (X_res, y_res) = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    if title is None:
        title = f'Resampling with {sampler.__class__.__name__}'
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)
import numpy as np

def plot_decision_function(X, y, clf, ax, title=None):
    if False:
        i = 10
        return i + 15
    plot_step = 0.02
    (x_min, x_max) = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    (y_min, y_max) = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    if title is not None:
        ax.set_title(title)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import ClusterCentroids
(X, y) = create_dataset(n_samples=400, weights=(0.05, 0.15, 0.8), class_sep=0.8)
samplers = {FunctionSampler(), ClusterCentroids(estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=0)}
(fig, axs) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, model, ax[0], title=f'Decision function with {sampler.__class__.__name__}')
    plot_resampling(X, y, sampler, ax[1])
fig.tight_layout()
from imblearn.under_sampling import RandomUnderSampler
(X, y) = create_dataset(n_samples=400, weights=(0.05, 0.15, 0.8), class_sep=0.8)
samplers = {FunctionSampler(), RandomUnderSampler(random_state=0)}
(fig, axs) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, model, ax[0], title=f'Decision function with {sampler.__class__.__name__}')
    plot_resampling(X, y, sampler, ax[1])
fig.tight_layout()
from imblearn.under_sampling import NearMiss
(X, y) = create_dataset(n_samples=1000, weights=(0.05, 0.15, 0.8), class_sep=1.5)
samplers = [NearMiss(version=1), NearMiss(version=2), NearMiss(version=3)]
(fig, axs) = plt.subplots(nrows=3, ncols=2, figsize=(15, 25))
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, model, ax[0], title=f'Decision function for {sampler.__class__.__name__}-{sampler.version}')
    plot_resampling(X, y, sampler, ax[1], title=f'Resampling using {sampler.__class__.__name__}-{sampler.version}')
fig.tight_layout()
from imblearn.under_sampling import AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
(X, y) = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)
samplers = [EditedNearestNeighbours(), RepeatedEditedNearestNeighbours(), AllKNN(allow_minority=True)]
(fig, axs) = plt.subplots(3, 2, figsize=(15, 25))
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, clf, ax[0], title=f'Decision function for \n{sampler.__class__.__name__}')
    plot_resampling(X, y, sampler, ax[1], title=f'Resampling using \n{sampler.__class__.__name__}')
fig.tight_layout()
from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule, OneSidedSelection
(X, y) = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)
(fig, axs) = plt.subplots(nrows=3, ncols=2, figsize=(15, 25))
samplers = [CondensedNearestNeighbour(random_state=0), OneSidedSelection(random_state=0), NeighbourhoodCleaningRule(n_neighbors=11)]
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, clf, ax[0], title=f'Decision function for \n{sampler.__class__.__name__}')
    plot_resampling(X, y, sampler, ax[1], title=f'Resampling using \n{sampler.__class__.__name__}')
fig.tight_layout()
from imblearn.under_sampling import InstanceHardnessThreshold
samplers = {FunctionSampler(), InstanceHardnessThreshold(estimator=LogisticRegression(), random_state=0)}
(fig, axs) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
for (ax, sampler) in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, model, ax[0], title=f'Decision function with \n{sampler.__class__.__name__}')
    plot_resampling(X, y, sampler, ax[1], title=f'Resampling using \n{sampler.__class__.__name__}')
fig.tight_layout()
plt.show()