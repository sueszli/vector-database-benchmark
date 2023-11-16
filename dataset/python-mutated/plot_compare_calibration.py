"""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of :term:`predict_proba` can be directly interpreted as a confidence level.
For instance, a well calibrated (binary) classifier should classify the samples
such that for the samples to which it gave a :term:`predict_proba` value close
to 0.8, approximately 80% actually belong to the positive class.

In this example we will compare the calibration of four different
models: :ref:`Logistic_regression`, :ref:`gaussian_naive_bayes`,
:ref:`Random Forest Classifier <forest>` and :ref:`Linear SVM
<svm_classification>`.

"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
(X, y) = make_classification(n_samples=100000, n_features=20, n_informative=2, n_redundant=2, random_state=42)
train_samples = 100
(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=False, test_size=100000 - train_samples)
import numpy as np
from sklearn.svm import LinearSVC

class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

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
            print('Hello World!')
        'Min-max scale output of `decision_function` to [0,1].'
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
lr = LogisticRegression()
gnb = GaussianNB()
svc = NaivelyCalibratedLinearSVC(C=1.0, dual='auto')
rfc = RandomForestClassifier()
clf_list = [(lr, 'Logistic'), (gnb, 'Naive Bayes'), (svc, 'SVC'), (rfc, 'Random forest')]
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap('Dark2')
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ['^', 'v', 's', 'o']
for (i, (clf, name)) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name=name, ax=ax_calibration_curve, color=colors(i), marker=markers[i])
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title('Calibration plots')
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for (i, (_, name)) in enumerate(clf_list):
    (row, col) = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])
    ax.hist(calibration_displays[name].y_prob, range=(0, 1), bins=10, label=name, color=colors(i))
    ax.set(title=name, xlabel='Mean predicted probability', ylabel='Count')
plt.tight_layout()
plt.show()