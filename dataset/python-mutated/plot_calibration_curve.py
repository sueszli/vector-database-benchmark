"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to
visualize how well calibrated the predicted probabilities are using calibration
curves, also known as reliability diagrams. Calibration of an uncalibrated
classifier will also be demonstrated.

"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
(X, y) = make_classification(n_samples=100000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.99, random_state=42)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
lr = LogisticRegression(C=1.0)
gnb = GaussianNB()
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method='isotonic')
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method='sigmoid')
clf_list = [(lr, 'Logistic'), (gnb, 'Naive Bayes'), (gnb_isotonic, 'Naive Bayes + Isotonic'), (gnb_sigmoid, 'Naive Bayes + Sigmoid')]
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap('Dark2')
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for (i, (clf, name)) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name=name, ax=ax_calibration_curve, color=colors(i))
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title('Calibration plots (Naive Bayes)')
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for (i, (_, name)) in enumerate(clf_list):
    (row, col) = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])
    ax.hist(calibration_displays[name].y_prob, range=(0, 1), bins=10, label=name, color=colors(i))
    ax.set(title=name, xlabel='Mean predicted probability', ylabel='Count')
plt.tight_layout()
plt.show()
from collections import defaultdict
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
scores = defaultdict(list)
for (i, (clf, name)) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores['Classifier'].append(name)
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        score_name = metric.__name__.replace('_', ' ').replace('score', '').capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))
    for metric in [precision_score, recall_score, f1_score]:
        score_name = metric.__name__.replace('_', ' ').replace('score', '').capitalize()
        scores[score_name].append(metric(y_test, y_pred))
    score_df = pd.DataFrame(scores).set_index('Classifier')
    score_df.round(decimals=3)
score_df
import numpy as np
from sklearn.svm import LinearSVC

class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output for binary classification."""

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Min-max scale output of `decision_function` to [0, 1].'
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba
lr = LogisticRegression(C=1.0)
svc = NaivelyCalibratedLinearSVC(max_iter=10000, dual='auto')
svc_isotonic = CalibratedClassifierCV(svc, cv=2, method='isotonic')
svc_sigmoid = CalibratedClassifierCV(svc, cv=2, method='sigmoid')
clf_list = [(lr, 'Logistic'), (svc, 'SVC'), (svc_isotonic, 'SVC + Isotonic'), (svc_sigmoid, 'SVC + Sigmoid')]
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for (i, (clf, name)) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name=name, ax=ax_calibration_curve, color=colors(i))
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title('Calibration plots (SVC)')
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for (i, (_, name)) in enumerate(clf_list):
    (row, col) = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])
    ax.hist(calibration_displays[name].y_prob, range=(0, 1), bins=10, label=name, color=colors(i))
    ax.set(title=name, xlabel='Mean predicted probability', ylabel='Count')
plt.tight_layout()
plt.show()
scores = defaultdict(list)
for (i, (clf, name)) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores['Classifier'].append(name)
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        score_name = metric.__name__.replace('_', ' ').replace('score', '').capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))
    for metric in [precision_score, recall_score, f1_score]:
        score_name = metric.__name__.replace('_', ' ').replace('score', '').capitalize()
        scores[score_name].append(metric(y_test, y_pred))
    score_df = pd.DataFrame(scores).set_index('Classifier')
    score_df.round(decimals=3)
score_df