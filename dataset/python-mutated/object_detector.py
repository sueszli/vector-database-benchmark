"""
Class definition and utilities for the object detection toolkit.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import time as _time
from datetime import datetime as _datetime
import six as _six
import turicreate as _tc
from turicreate.toolkits._model import Model as _Model
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits import _coreml_utils
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe, _numeric_param_check_range, _raise_error_if_not_iterable
from turicreate.toolkits._main import ToolkitError as _ToolkitError
from turicreate._deps.minimal_package import _minimal_package_import_check
from .. import _pre_trained_models
from .._mps_utils import MpsGraphAPI as _MpsGraphAPI, MpsGraphNetworkType as _MpsGraphNetworkType

def _get_mps_od_net(input_image_shape, batch_size, output_size, anchors, config, weights={}):
    if False:
        while True:
            i = 10
    '\n    Initializes an MpsGraphAPI for object detection.\n    '
    network = _MpsGraphAPI(network_id=_MpsGraphNetworkType.kODGraphNet)
    (c_in, h_in, w_in) = input_image_shape
    c_out = output_size
    h_out = h_in // 32
    w_out = w_in // 32
    network.init(batch_size, c_in, h_in, w_in, c_out, h_out, w_out, weights=weights, config=config)
    return network

def _raise_error_if_not_detection_sframe(dataset, feature, annotations, require_annotations):
    if False:
        for i in range(10):
            print('nop')
    _raise_error_if_not_sframe(dataset, 'datset')
    if feature not in dataset.column_names():
        raise _ToolkitError("Feature column '%s' does not exist" % feature)
    if dataset[feature].dtype != _tc.Image:
        raise _ToolkitError('Feature column must contain images')
    if require_annotations:
        if annotations not in dataset.column_names():
            raise _ToolkitError("Annotations column '%s' does not exist" % annotations)
        if dataset[annotations].dtype not in [list, dict]:
            raise _ToolkitError('Annotations column must be of type dict or list')

def create(dataset, annotations=None, feature=None, model='darknet-yolo', classes=None, batch_size=0, max_iterations=0, verbose=True, grid_shape=[13, 13], random_seed=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a :class:`ObjectDetector` model.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Input data. The columns named by the ``feature`` and ``annotations``\n        parameters will be extracted for training the detector.\n\n    annotations : string\n        Name of the column containing the object detection annotations.  This\n        column should be a list of dictionaries (or a single dictionary), with\n        each dictionary representing a bounding box of an object instance. Here\n        is an example of the annotations for a single image with two object\n        instances::\n\n            [{\'label\': \'dog\',\n              \'type\': \'rectangle\',\n              \'coordinates\': {\'x\': 223, \'y\': 198,\n                              \'width\': 130, \'height\': 230}},\n             {\'label\': \'cat\',\n              \'type\': \'rectangle\',\n              \'coordinates\': {\'x\': 40, \'y\': 73,\n                              \'width\': 80, \'height\': 123}}]\n\n        The value for `x` is the horizontal center of the box paired with\n        `width` and `y` is the vertical center of the box paired with `height`.\n        \'None\' (the default) indicates the only list column in `dataset` should\n        be used for the annotations.\n\n    feature : string\n        Name of the column containing the input images. \'None\' (the default)\n        indicates the only image column in `dataset` should be used as the\n        feature.\n\n    model : string optional\n        Object detection model to use:\n\n           - "darknet-yolo" : Fast and medium-sized model\n\n    grid_shape : array optional\n        Shape of the grid used for object detection. Higher values increase precision for small objects, but at a higher computational cost\n\n           - [13, 13] : Default grid value for a Fast and medium-sized model\n\n    classes : list optional\n        List of strings containing the names of the classes of objects.\n        Inferred from the data if not provided.\n\n    batch_size: int\n        The number of images per training iteration. If 0, then it will be\n        automatically determined based on resource availability.\n\n    max_iterations : int\n        The number of training iterations. If 0, then it will be automatically\n        be determined based on the amount of data you provide.\n\n    random_seed : int, optional\n        The results can be reproduced when given the same seed.\n\n    verbose : bool, optional\n        If True, print progress updates and model details.\n\n    Returns\n    -------\n    out : ObjectDetector\n        A trained :class:`ObjectDetector` model.\n\n    See Also\n    --------\n    ObjectDetector\n\n    Examples\n    --------\n    .. sourcecode:: python\n\n        # Train an object detector model\n        >>> model = turicreate.object_detector.create(data)\n\n        # Make predictions on the training set and as column to the SFrame\n        >>> data[\'predictions\'] = model.predict(data)\n\n        # Visualize predictions by generating a new column of marked up images\n        >>> data[\'image_pred\'] = turicreate.object_detector.util.draw_bounding_boxes(data[\'image\'], data[\'predictions\'])\n    '
    _raise_error_if_not_sframe(dataset, 'dataset')
    if len(dataset) == 0:
        raise _ToolkitError('Unable to train on empty dataset')
    _numeric_param_check_range('max_iterations', max_iterations, 0, _six.MAXSIZE)
    start_time = _time.time()
    supported_detectors = ['darknet-yolo']
    if feature is None:
        feature = _tkutl._find_only_image_column(dataset)
        if verbose:
            print("Using '%s' as feature column" % feature)
    if annotations is None:
        annotations = _tkutl._find_only_column_of_type(dataset, target_type=[list, dict], type_name='list', col_name='annotations')
        if verbose:
            print("Using '%s' as annotations column" % annotations)
    _raise_error_if_not_detection_sframe(dataset, feature, annotations, require_annotations=True)
    _tkutl._handle_missing_values(dataset, feature, 'dataset')
    _tkutl._check_categorical_option_type('model', model, supported_detectors)
    base_model = model.split('-', 1)[0]
    ref_model = _pre_trained_models.OBJECT_DETECTION_BASE_MODELS[base_model]()
    pretrained_model = _pre_trained_models.OBJECT_DETECTION_BASE_MODELS['darknet_mlmodel']()
    pretrained_model_path = pretrained_model.get_model_path()
    params = {'anchors': [(1.0, 2.0), (1.0, 1.0), (2.0, 1.0), (2.0, 4.0), (2.0, 2.0), (4.0, 2.0), (4.0, 8.0), (4.0, 4.0), (8.0, 4.0), (8.0, 16.0), (8.0, 8.0), (16.0, 8.0), (16.0, 32.0), (16.0, 16.0), (32.0, 16.0)], 'grid_shape': grid_shape, 'aug_resize': 0, 'aug_rand_crop': 0.9, 'aug_rand_pad': 0.9, 'aug_rand_gray': 0.0, 'aug_aspect_ratio': 1.25, 'aug_hue': 0.05, 'aug_brightness': 0.05, 'aug_saturation': 0.05, 'aug_contrast': 0.05, 'aug_horizontal_flip': True, 'aug_min_object_covered': 0, 'aug_min_eject_coverage': 0.5, 'aug_area_range': (0.15, 2), 'aug_pca_noise': 0.0, 'aug_max_attempts': 20, 'aug_inter_method': 2, 'lmb_coord_xy': 10.0, 'lmb_coord_wh': 10.0, 'lmb_obj': 100.0, 'lmb_noobj': 5.0, 'lmb_class': 2.0, 'non_maximum_suppression_threshold': 0.45, 'rescore': True, 'clip_gradients': 0.025, 'weight_decay': 0.0005, 'sgd_momentum': 0.9, 'learning_rate': 0.001, 'shuffle': True, 'mps_loss_mult': 8, 'io_thread_buffer_size': 8, 'mlmodel_path': pretrained_model_path}
    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
    if classes == None:
        classes = []
    _raise_error_if_not_iterable(classes)
    _raise_error_if_not_iterable(grid_shape)
    grid_shape = [int(x) for x in grid_shape]
    assert len(grid_shape) == 2
    tf_config = {'grid_height': params['grid_shape'][0], 'grid_width': params['grid_shape'][1], 'mlmodel_path': params['mlmodel_path'], 'classes': classes, 'compute_final_metrics': False, 'verbose': verbose, 'model': 'darknet-yolo', 'random_seed': random_seed}
    if batch_size > 0:
        tf_config['batch_size'] = batch_size
    if max_iterations > 0:
        tf_config['max_iterations'] = max_iterations
    model = _tc.extensions.object_detector()
    model.train(data=dataset, annotations_column_name=annotations, image_column_name=feature, options=tf_config)
    return ObjectDetector(model_proxy=model, name='object_detector')

class ObjectDetector(_Model):
    """
    A trained model using C++ implementation that is ready to use for classification
    or export to CoreML.

    This model should not be constructed directly.
    """
    _CPP_OBJECT_DETECTOR_VERSION = 1

    def __init__(self, model_proxy=None, name=None):
        if False:
            return 10
        self.__proxy__ = model_proxy
        self.__name__ = name

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return 'object_detector'

    def __str__(self):
        if False:
            return 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the ObjectDetector.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        width = 40
        (sections, section_titles) = self._get_summary_struct()
        out = _tkutl._toolkit_repr_print(self, sections, section_titles, width=width)
        return out

    def _get_version(self):
        if False:
            i = 10
            return i + 15
        return self._CPP_OBJECT_DETECTOR_VERSION

    def export_coreml(self, filename, include_non_maximum_suppression=True, iou_threshold=None, confidence_threshold=None):
        if False:
            while True:
                i = 10
        "\n        Save the model in Core ML format. The Core ML model takes an image of\n        fixed size as input and produces two output arrays: `confidence` and\n        `coordinates`.\n\n        The first one, `confidence` is an `N`-by-`C` array, where `N` is the\n        number of instances predicted and `C` is the number of classes. The\n        number `N` is fixed and will include many low-confidence predictions.\n        The instances are not sorted by confidence, so the first one will\n        generally not have the highest confidence (unlike in `predict`). Also\n        unlike the `predict` function, the instances have not undergone\n        what is called `non-maximum suppression`, which means there could be\n        several instances close in location and size that have all discovered\n        the same object instance. Confidences do not need to sum to 1 over the\n        classes; any remaining probability is implied as confidence there is no\n        object instance present at all at the given coordinates. The classes\n        appear in the array alphabetically sorted.\n\n        The second array `coordinates` is of size `N`-by-4, where the first\n        dimension `N` again represents instances and corresponds to the\n        `confidence` array. The second dimension represents `x`, `y`, `width`,\n        `height`, in that order.  The values are represented in relative\n        coordinates, so (0.5, 0.5) represents the center of the image and (1,\n        1) the bottom right corner. You will need to multiply the relative\n        values with the original image size before you resized it to the fixed\n        input size to get pixel-value coordinates similar to `predict`.\n\n        See Also\n        --------\n        save\n\n        Parameters\n        ----------\n        filename : string\n            The path of the file where we want to save the Core ML model.\n\n        include_non_maximum_suppression : bool\n            Non-maximum suppression is only available in iOS 12+.\n            A boolean parameter to indicate whether the Core ML model should be\n            saved with built-in non-maximum suppression or not.\n            This parameter is set to True by default.\n\n        iou_threshold : float\n            Threshold value for non-maximum suppression. Non-maximum suppression\n            prevents multiple bounding boxes appearing over a single object.\n            This threshold, set between 0 and 1, controls how aggressive this\n            suppression is. A value of 1 means no maximum suppression will\n            occur, while a value of 0 will maximally suppress neighboring\n            boxes around a prediction.\n\n        confidence_threshold : float\n            Only return predictions above this level of confidence. The\n            threshold can range from 0 to 1.\n\n        Examples\n        --------\n        >>> model.export_coreml('detector.mlmodel')\n        "
        options = {}
        options['include_non_maximum_suppression'] = include_non_maximum_suppression
        options['version'] = self._get_version()
        if confidence_threshold is not None:
            options['confidence_threshold'] = confidence_threshold
        if iou_threshold is not None:
            options['iou_threshold'] = iou_threshold
        additional_user_defined_metadata = _coreml_utils._get_tc_version_info()
        short_description = _coreml_utils._mlmodel_short_description('Object Detector')
        self.__proxy__.export_to_coreml(filename, short_description, additional_user_defined_metadata, options)

    def predict(self, dataset, confidence_threshold=0.25, iou_threshold=0.45, verbose=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Predict object instances in an SFrame of images.\n\n        Parameters\n        ----------\n        dataset : SFrame | SArray | turicreate.Image\n            The images on which to perform object detection.\n            If dataset is an SFrame, it must have a column with the same name\n            as the feature column during training. Additional columns are\n            ignored.\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions. Each element corresponds to\n            an image and contains a list of dictionaries. Each dictionary\n            describes an object instances that was found in the image. If\n            `dataset` is a single image, the return value will be a single\n            prediction.\n\n        See Also\n        --------\n        evaluate\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            # Make predictions\n            >>> pred = model.predict(data)\n\n            # Stack predictions, for a better overview\n            >>> turicreate.object_detector.util.stack_annotations(pred)\n            Data:\n            +--------+------------+-------+-------+-------+-------+--------+\n            | row_id | confidence | label |   x   |   y   | width | height |\n            +--------+------------+-------+-------+-------+-------+--------+\n            |   0    |    0.98    |  dog  | 123.0 | 128.0 |  80.0 | 182.0  |\n            |   0    |    0.67    |  cat  | 150.0 | 183.0 | 129.0 | 101.0  |\n            |   1    |    0.8     |  dog  |  50.0 | 432.0 |  65.0 |  98.0  |\n            +--------+------------+-------+-------+-------+-------+--------+\n            [3 rows x 7 columns]\n\n            # Visualize predictions by generating a new column of marked up images\n            >>> data['image_pred'] = turicreate.object_detector.util.draw_bounding_boxes(data['image'], data['predictions'])\n        "
        _numeric_param_check_range('confidence_threshold', confidence_threshold, 0.0, 1.0)
        _numeric_param_check_range('iou_threshold', iou_threshold, 0.0, 1.0)
        options = {}
        options['confidence_threshold'] = confidence_threshold
        options['iou_threshold'] = iou_threshold
        options['verbose'] = verbose
        return self.__proxy__.predict(dataset, options)

    def evaluate(self, dataset, metric='auto', output_type='dict', confidence_threshold=0.001, iou_threshold=0.45):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate the model by making predictions and comparing these to ground\n        truth bounding box annotations.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the annotations and feature used for model training.\n            Additional columns are ignored.\n\n        metric : str or list, optional\n            Name of the evaluation metric or list of several names. The primary\n            metric is average precision, which is the area under the\n            precision/recall curve and reported as a value between 0 and 1 (1\n            being perfect). Possible values are:\n\n            - 'auto'                      : Returns all primary metrics.\n            - 'all'                       : Returns all available metrics.\n            - 'average_precision_50'      : Average precision per class with\n                                            intersection-over-union threshold at\n                                            50% (PASCAL VOC metric).\n            - 'average_precision'         : Average precision per class calculated over multiple\n                                            intersection-over-union thresholds\n                                            (at 50%, 55%, ..., 95%) and averaged.\n            - 'mean_average_precision_50' : Mean over all classes (for ``'average_precision_50'``).\n                                            This is the primary single-value metric.\n            - 'mean_average_precision'    : Mean over all classes (for ``'average_precision'``)\n\n        Returns\n        -------\n        out : dict / SFrame\n            Output type depends on the option `output_type`.\n\n        See Also\n        --------\n        create, predict\n\n        Examples\n        --------\n        >>> results = model.evaluate(data)\n        >>> print('mAP: {:.1%}'.format(results['mean_average_precision']))\n        mAP: 43.2%\n        "
        _numeric_param_check_range('confidence_threshold', confidence_threshold, 0.0, 1.0)
        _numeric_param_check_range('iou_threshold', iou_threshold, 0.0, 1.0)
        options = {}
        options['confidence_threshold'] = confidence_threshold
        options['iou_threshold'] = iou_threshold
        return self.__proxy__.evaluate(dataset, metric, output_type, options)

    def _get_summary_struct(self):
        if False:
            print('Hello World!')
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Model', 'model'), ('Number of classes', 'num_classes'), ('Input image shape', 'input_image_shape')]
        training_fields = [('Training time', '_training_time_as_string'), ('Training epochs', 'training_epochs'), ('Training iterations', 'training_iterations'), ('Number of examples (images)', 'num_examples'), ('Number of bounding boxes (instances)', 'num_bounding_boxes'), ('Final loss (specific to model)', 'training_loss')]
        section_titles = ['Schema', 'Training summary']
        return ([model_fields, training_fields], section_titles)