from skimage.feature import multiscale_basic_features
try:
    from sklearn.exceptions import NotFittedError
    from sklearn.ensemble import RandomForestClassifier
    has_sklearn = True
except ImportError:
    has_sklearn = False

    class NotFittedError(Exception):
        pass

class TrainableSegmenter:
    """Estimator for classifying pixels.

    Parameters
    ----------
    clf : classifier object, optional
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        ``(m_features, *labels.shape)``. If None,
        :func:`skimage.feature.multiscale_basic_features` is used.

    Methods
    -------
    compute_features
    fit
    predict
    """

    def __init__(self, clf=None, features_func=None):
        if False:
            return 10
        if clf is None:
            if has_sklearn:
                self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            else:
                raise ImportError('Please install scikit-learn or pass a classifier instanceto TrainableSegmenter.')
        else:
            self.clf = clf
        self.features_func = features_func

    def compute_features(self, image):
        if False:
            for i in range(10):
                print('nop')
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        self.features = self.features_func(image)

    def fit(self, image, labels):
        if False:
            for i in range(10):
                print('nop')
        'Train classifier using partially labeled (annotated) image.\n\n        Parameters\n        ----------\n        image : ndarray\n            Input image, which can be grayscale or multichannel, and must have a\n            number of dimensions compatible with ``self.features_func``.\n        labels : ndarray of ints\n            Labeled array of shape compatible with ``image`` (same shape for a\n            single-channel image). Labels >= 1 correspond to the training set and\n            label 0 to unlabeled pixels to be segmented.\n        '
        self.compute_features(image)
        fit_segmenter(labels, self.features, self.clf)

    def predict(self, image):
        if False:
            return 10
        'Segment new image using trained internal classifier.\n\n        Parameters\n        ----------\n        image : ndarray\n            Input image, which can be grayscale or multichannel, and must have a\n            number of dimensions compatible with ``self.features_func``.\n\n        Raises\n        ------\n        NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).\n        '
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        features = self.features_func(image)
        return predict_segmenter(features, self.clf)

def fit_segmenter(labels, features, clf):
    if False:
        return 10
    "Segmentation using labeled parts of the image and a classifier.\n\n    Parameters\n    ----------\n    labels : ndarray of ints\n        Image of labels. Labels >= 1 correspond to the training set and\n        label 0 to unlabeled pixels to be segmented.\n    features : ndarray\n        Array of features, with the first dimension corresponding to the number\n        of features, and the other dimensions correspond to ``labels.shape``.\n    clf : classifier object\n        classifier object, exposing a ``fit`` and a ``predict`` method as in\n        scikit-learn's API, for example an instance of\n        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.\n\n    Returns\n    -------\n    clf : classifier object\n        classifier trained on ``labels``\n\n    Raises\n    ------\n    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).\n    "
    mask = labels > 0
    training_data = features[mask]
    training_labels = labels[mask].ravel()
    clf.fit(training_data, training_labels)
    return clf

def predict_segmenter(features, clf):
    if False:
        while True:
            i = 10
    "Segmentation of images using a pretrained classifier.\n\n    Parameters\n    ----------\n    features : ndarray\n        Array of features, with the last dimension corresponding to the number\n        of features, and the other dimensions are compatible with the shape of\n        the image to segment, or a flattened image.\n    clf : classifier object\n        trained classifier object, exposing a ``predict`` method as in\n        scikit-learn's API, for example an instance of\n        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The\n        classifier must be already trained, for example with\n        :func:`skimage.future.fit_segmenter`.\n\n    Returns\n    -------\n    output : ndarray\n        Labeled array, built from the prediction of the classifier.\n    "
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))
    try:
        predicted_labels = clf.predict(features)
    except NotFittedError:
        raise NotFittedError('You must train the classifier `clf` firstfor example with the `fit_segmenter` function.')
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(err.args[0] + '\n' + 'Maybe you did not use the same type of features for training the classifier.')
        else:
            raise err
    output = predicted_labels.reshape(sh[:-1])
    return output