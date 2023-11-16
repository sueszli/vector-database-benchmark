"""
==========================================
IsolationForest benchmark
==========================================
A test of IsolationForest on classical anomaly detection datasets.

The benchmark is run as follows:
1. The dataset is randomly split into a training set and a test set, both
assumed to contain outliers.
2. Isolation Forest is trained on the training set.
3. The ROC curve is computed on the test set using the knowledge of the labels.

Note that the smtp dataset contains a very small proportion of outliers.
Therefore, depending on the seed of the random number generator, randomly
splitting the data set might lead to a test set containing no outliers. In this
case a warning is raised when computing the ROC curve.
"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle as sh
print(__doc__)

def print_outlier_ratio(y):
    if False:
        while True:
            i = 10
    '\n    Helper function to show the distinct value count of element in the target.\n    Useful indicator for the datasets used in bench_isolation_forest.py.\n    '
    (uniq, cnt) = np.unique(y, return_counts=True)
    print('----- Target count values: ')
    for (u, c) in zip(uniq, cnt):
        print('------ %s -> %d occurrences' % (str(u), c))
    print('----- Outlier ratio: %.5f' % (np.min(cnt) / len(y)))
random_state = 1
(fig_roc, ax_roc) = plt.subplots(1, 1, figsize=(8, 5))
with_decision_function_histograms = False
datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
for dat in datasets:
    print('====== %s ======' % dat)
    print('--- Fetching data...')
    if dat in ['http', 'smtp', 'SF', 'SA']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=True, random_state=random_state)
        X = dataset.data
        y = dataset.target
    if dat == 'shuttle':
        dataset = fetch_openml('shuttle', as_frame=False, parser='pandas')
        X = dataset.data
        y = dataset.target.astype(np.int64)
        (X, y) = sh(X, y, random_state=random_state)
        s = y != 4
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)
        print('----- ')
    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True, random_state=random_state)
        X = dataset.data
        y = dataset.target
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)
        print_outlier_ratio(y)
    print('--- Vectorizing data...')
    if dat == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)
    if dat == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)
    if dat in ('http', 'smtp'):
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)
    (n_samples, n_features) = X.shape
    n_samples_train = n_samples // 2
    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]
    print('--- Fitting the IsolationForest estimator...')
    model = IsolationForest(n_jobs=-1, random_state=random_state)
    tstart = time()
    model.fit(X_train)
    fit_time = time() - tstart
    tstart = time()
    scoring = -model.decision_function(X_test)
    print('--- Preparing the plot elements...')
    if with_decision_function_histograms:
        (fig, ax) = plt.subplots(3, sharex=True, sharey=True)
        bins = np.linspace(-0.5, 0.5, 200)
        ax[0].hist(scoring, bins, color='black')
        ax[0].set_title('Decision function for %s dataset' % dat)
        ax[1].hist(scoring[y_test == 0], bins, color='b', label='normal data')
        ax[1].legend(loc='lower right')
        ax[2].hist(scoring[y_test == 1], bins, color='r', label='outliers')
        ax[2].legend(loc='lower right')
    predict_time = time() - tstart
    (fpr, tpr, thresholds) = roc_curve(y_test, scoring)
    auc_score = auc(fpr, tpr)
    label = '%s (AUC: %0.3f, train_time= %0.2fs, test_time= %0.2fs)' % (dat, auc_score, fit_time, predict_time)
    print(label)
    ax_roc.plot(fpr, tpr, lw=1, label=label)
ax_roc.set_xlim([-0.05, 1.05])
ax_roc.set_ylim([-0.05, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver operating characteristic (ROC) curves')
ax_roc.legend(loc='lower right')
fig_roc.tight_layout()
plt.show()