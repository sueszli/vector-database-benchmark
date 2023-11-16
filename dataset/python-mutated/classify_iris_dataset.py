__author__ = 'nastra'
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from matplotlib import pyplot as plt
from sklearn.neighbors.nearest_centroid import NearestCentroid

def plot_iris(features, feature_names):
    if False:
        for i in range(10):
            print('nop')
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for (i, (p0, p1)) in enumerate(pairs):
        plt.subplot(2, 3, i + 1)
        for (t, marker, c) in zip(range(3), '>ox', 'rgb'):
            plt.scatter(features[target == t, p0], features[target == t, p1], marker=marker, c=c)
        plt.xlabel(feature_names[p0])
        plt.ylabel(feature_names[p1])
        plt.xticks([])
        plt.yticks([])
    plt.show()
data = datasets.load_iris()
features = data['data']
labels = data['target_names'][data['target']]
feature_names = data['feature_names']
target = data.target
plot_iris(features, feature_names)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
from sklearn.cross_validation import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(features, target, test_size=0.2, random_state=42)
h = 0.02
weights = 'distance'
classifier = neighbors.KNeighborsClassifier(5, weights='uniform')
classifier.fit(X_train, y_train)
from sklearn import svm
svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)
from sklearn import cross_validation
scores = cross_validation.cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
svm_scores = cross_validation.cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='f1')
test_scores = cross_validation.cross_val_score(classifier, X_test, y_test, cv=5, scoring='f1')
svm_test_scores = cross_validation.cross_val_score(svm_classifier, X_test, y_test, cv=5, scoring='f1')
print('Training Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print('Test Accuracy for KNN: %0.2f (+/- %0.2f)' % (test_scores.mean(), test_scores.std() * 2))
print('Training Accuracy: %0.2f (+/- %0.2f)' % (svm_scores.mean(), svm_scores.std() * 2))
print('Test Accuracy for SVM: %0.2f (+/- %0.2f)' % (svm_test_scores.mean(), svm_test_scores.std() * 2))