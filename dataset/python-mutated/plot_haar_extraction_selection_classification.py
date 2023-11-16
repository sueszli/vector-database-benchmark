"""
======================================================
Face classification using Haar-like feature descriptor
======================================================

Haar-like feature descriptors were successfully used to implement the first
real-time face detector [1]_. Inspired by this application, we propose an
example illustrating the extraction, selection, and classification of Haar-like
features to detect faces vs. non-faces.

Notes
-----

This example relies on `scikit-learn <https://scikit-learn.org/>`_ for feature
selection and classification.

References
----------

.. [1] Viola, Paul, and Michael J. Jones. "Robust real-time face
       detection." International journal of computer vision 57.2
       (2004): 137-154.
       https://www.merl.com/publications/docs/TR2004-043.pdf
       :DOI:`10.1109/CVPR.2001.990517`

"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

def extract_feature_image(img, feature_type, feature_coord=None):
    if False:
        return 10
    'Extract the haar feature for the current image'
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=feature_type, feature_coord=feature_coord)
images = lfw_subset()
feature_types = ['type-2-x', 'type-2-y']
t_start = time()
X = [extract_feature_image(img, feature_types) for img in images]
X = np.stack(X)
time_full_feature_comp = time() - t_start
y = np.array([1] * 100 + [0] * 100)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=150, random_state=0, stratify=y)
(feature_coord, feature_type) = haar_like_feature_coord(width=images.shape[2], height=images.shape[1], feature_type=feature_types)
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
idx_sorted = np.argsort(clf.feature_importances_)[::-1]
(fig, axes) = plt.subplots(3, 2)
for (idx, ax) in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0, images.shape[2], images.shape[1], [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
_ = fig.suptitle('The most important features')
cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
cdf_feature_importances /= cdf_feature_importances[-1]
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
sig_feature_percent = round(sig_feature_count / len(cdf_feature_importances) * 100, 1)
print(f'{sig_feature_count} features, or {sig_feature_percent}%, account for 70% of branch points in the random forest.')
feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
t_start = time()
X = [extract_feature_image(img, feature_type_sel, feature_coord_sel) for img in images]
X = np.stack(X)
time_subs_feature_comp = time() - t_start
y = np.array([1] * 100 + [0] * 100)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=150, random_state=0, stratify=y)
t_start = time()
clf.fit(X_train, y_train)
time_subs_train = time() - t_start
auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
summary = f'Computing the full feature set took {time_full_feature_comp:.3f}s, plus {time_full_train:.3f}s training, for an AUC of {auc_full_features:.2f}. Computing the restricted feature set took {time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s training, for an AUC of {auc_subs_features:.2f}.'
print(summary)
plt.show()