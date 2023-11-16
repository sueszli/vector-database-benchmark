"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm

class ImageExplanation(object):

    def __init__(self, image, segments):
        if False:
            return 10
        'Init function.\n\n        Args:\n            image: 3d numpy array\n            segments: 2d numpy array, with the output from skimage.segmentation\n        '
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0):
        if False:
            while True:
                i = 10
        'Init function.\n\n        Args:\n            label: label to explain\n            positive_only: if True, only take superpixels that contribute to\n                the prediction of the label. Otherwise, use the top\n                num_features superpixels, which can be positive or negative\n                towards the label\n            hide_rest: if True, make the non-explanation part of the return\n                image gray\n            num_features: number of superpixels to include in explanation\n            min_weight: TODO\n\n        Returns:\n            (image, mask), where image is a 3d numpy array and mask is a 2d\n            numpy array that can be used with\n            skimage.segmentation.mark_boundaries\n        '
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return (temp, mask)
        else:
            for (f, w) in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
            return (temp, mask)

class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None):
        if False:
            while True:
                i = 10
        "Init function.\n\n        Args:\n            kernel_width: kernel width for the exponential kernel.\n            If None, defaults to sqrt(number of columns) * 0.75.\n            kernel: similarity kernel that takes euclidean distances and kernel\n                width as input and outputs weights in (0,1). If None, defaults to\n                an exponential kernel.\n            verbose: if true, print local prediction values from linear model\n            feature_selection: feature selection method. can be\n                'forward_selection', 'lasso_path', 'none' or 'auto'.\n                See function 'explain_instance_with_data' in lime_base.py for\n                details on what each of the options does.\n            random_state: an integer or numpy.RandomState that will be used to\n                generate random numbers. If None, the random state will be\n                initialized using the internal numpy seed.\n        "
        kernel_width = float(kernel_width)
        if kernel is None:

            def kernel(d, kernel_width):
                if False:
                    while True:
                        i = 10
                return np.sqrt(np.exp(-d ** 2 / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,), hide_color=None, top_labels=5, num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None):
        if False:
            i = 10
            return i + 15
        "Generates explanations for a prediction.\n\n        First, we generate neighborhood data by randomly perturbing features\n        from the instance (see __data_inverse). We then learn locally weighted\n        linear models on this neighborhood data to explain each of the classes\n        in an interpretable way (see lime_base.py).\n\n        Args:\n            image: 3 dimension RGB image. If this is only two dimensional,\n                we will assume it's a grayscale image and call gray2rgb.\n            classifier_fn: classifier prediction probability function, which\n                takes a numpy array and outputs prediction probabilities.  For\n                ScikitClassifiers , this is classifier.predict_proba.\n            labels: iterable with labels to be explained.\n            hide_color: TODO\n            top_labels: if not None, ignore labels and produce explanations for\n                the K labels with highest prediction probabilities, where K is\n                this parameter.\n            num_features: maximum number of features present in explanation\n            num_samples: size of the neighborhood to learn the linear model\n            batch_size: TODO\n            distance_metric: the distance metric to use for weights.\n            model_regressor: sklearn regressor to use in explanation. Defaults\n            to Ridge regression in LimeBase. Must have model_regressor.coef_\n            and 'sample_weight' as a parameter to model_regressor.fit()\n            segmentation_fn: SegmentationAlgorithm, wrapped skimage\n            segmentation function\n            random_seed: integer used as random seed for the segmentation\n                algorithm. If None, a random integer, between 0 and 1000,\n                will be generated using the internal random number generator.\n\n        Returns:\n            An Explanation object (see explanation.py) with the corresponding\n            explanations.\n        "
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2, random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (np.mean(image[segments == x][:, 0]), np.mean(image[segments == x][:, 1]), np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color
        top = labels
        (data, labels) = self.data_labels(image, fudged_image, segments, classifier_fn, num_samples, batch_size=batch_size)
        distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric=distance_metric).ravel()
        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label], ret_exp.local_exp[label], ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(data, labels, distances, label, num_features, model_regressor=model_regressor, feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self, image, fudged_image, segments, classifier_fn, num_samples, batch_size=10):
        if False:
            for i in range(10):
                print('nop')
        'Generates images and predictions in the neighborhood of this image.\n\n        Args:\n            image: 3d numpy array, the image\n            fudged_image: 3d numpy array, image to replace original image when\n                superpixel is turned off\n            segments: segmentation of the image\n            classifier_fn: function that takes a list of images and returns a\n                matrix of prediction probabilities\n            num_samples: size of the neighborhood to learn the linear model\n            batch_size: classifier_fn will be called on batches of this size.\n\n        Returns:\n            A tuple (data, labels), where:\n                data: dense num_samples * num_superpixels\n                labels: prediction probabilities matrix\n        '
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return (data, np.array(labels))