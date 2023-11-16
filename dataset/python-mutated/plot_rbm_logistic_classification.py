"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""
import numpy as np
from scipy.ndimage import convolve
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

def nudge_dataset(X, Y):
    if False:
        for i in range(10):
            print('nop')
    '\n    This produces a dataset 5 times bigger than the original one,\n    by moving the 8x8 images in X around by 1px to left, right, down, up\n    '
    direction_vectors = [[[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]

    def shift(x, w):
        if False:
            return 10
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return (X, Y)
(X, y) = datasets.load_digits(return_X_y=True)
X = np.asarray(X, 'float32')
(X, Y) = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
from sklearn.base import clone
rbm.learning_rate = 0.06
rbm.n_iter = 10
rbm.n_components = 100
logistic.C = 6000
rbm_features_classifier.fit(X_train, Y_train)
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)
from sklearn import metrics
Y_pred = rbm_features_classifier.predict(X_test)
print('Logistic regression using RBM features:\n%s\n' % metrics.classification_report(Y_test, Y_pred))
Y_pred = raw_pixel_classifier.predict(X_test)
print('Logistic regression using raw pixel features:\n%s\n' % metrics.classification_report(Y_test, Y_pred))
import matplotlib.pyplot as plt
plt.figure(figsize=(4.2, 4))
for (i, comp) in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()