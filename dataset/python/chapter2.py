__author__ = 'nastra'

from matplotlib import pyplot as plt
from sklearn import datasets as ds

# load data from scikit learn
data = ds.load_iris()
labels = data['target_names'][data['target']]

features = data['data']
feature_names = data['feature_names']
target = data['target']

pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i,(p0,p1) in enumerate(pairs):
    plt.subplot(2,3,i+1)
    for t,marker,c in zip(range(3),">ox","rgb"):
        plt.scatter(features[target == t,p0], features[target == t,p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])

plt.show()


# show setosa
plength = features[:,2]
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('maximum of setosa: {0}.'.format(max_setosa))
print('minimum of others: {0}.'.format(min_non_setosa))

non_setosa_features = features[~is_setosa]
non_setosa_labels = labels[~is_setosa]
virginica = (labels == 'virginica')

best_acc = -1.0
for fi in range(features.shape[1]):
    thresh = features[:,fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
print('Best cut is {0} on feature {1}, which achieves accuracy of {2:.1%}.'.format(best_t,best_fi,best_acc))