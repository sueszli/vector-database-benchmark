import turicreate as _tc
from turicreate.toolkits._model import CustomModel as _CustomModel
from turicreate.toolkits._model import PythonProxy as _PythonProxy, ToolkitError as _ToolkitError
from turicreate.toolkits.object_detector.object_detector import ObjectDetector as _ObjectDetector
from turicreate.toolkits.one_shot_object_detector.util._augmentation import preview_synthetic_training_data as _preview_synthetic_training_data
import turicreate.toolkits._internal_utils as _tkutl

def create(data, target, backgrounds=None, batch_size=0, max_iterations=0, verbose=True):
    if False:
        i = 10
        return i + 15
    "\n    Create a :class:`OneShotObjectDetector` model. Note: The One Shot Object Detector\n    is currently in beta.\n\n    Parameters\n    ----------\n    data : SFrame | tc.Image\n        A single starter image or an SFrame that contains the starter images\n        along with their corresponding labels.  These image(s) can be in either\n        RGB or RGBA format. They should not be padded.\n\n    target : string\n        Name of the target (when data is a single image) or the target column\n        name (when data is an SFrame of images).\n\n    backgrounds : optional SArray\n        A list of backgrounds used for synthetic data generation. When set to\n        None, a set of default backgrounds are downloaded and used.\n\n    batch_size : int\n        The number of images per training iteration. If 0, then it will be\n        automatically determined based on resource availability.\n\n    max_iterations : int\n        The number of training iterations. If 0, then it will be automatically\n        be determined based on the amount of data you provide.\n\n    verbose : bool optional\n        If True, print progress updates and model details.\n\n    Examples\n    --------\n    .. sourcecode:: python\n\n        # Train an object detector model\n        >>> model = turicreate.one_shot_object_detector.create(train_data, label = 'cards')\n\n        # Make predictions on the training set and as column to the SFrame\n        >>> test_data['predictions'] = model.predict(test_data)\n    "
    if not isinstance(data, _tc.SFrame) and (not isinstance(data, _tc.Image)):
        raise TypeError("'data' must be of type SFrame or tc.Image.")
    if isinstance(data, _tc.SFrame) and len(data) == 0:
        raise _ToolkitError("'data' can not be an empty SFrame")
    augmented_data = _preview_synthetic_training_data(data, target, backgrounds)
    model = _tc.object_detector.create(augmented_data, batch_size=batch_size, max_iterations=max_iterations, verbose=verbose)
    if isinstance(data, _tc.SFrame):
        num_starter_images = len(data)
    else:
        num_starter_images = 1
    state = {'detector': model, 'target': target, 'num_classes': model.num_classes, 'num_starter_images': num_starter_images, '_detector_version': _ObjectDetector._CPP_OBJECT_DETECTOR_VERSION}
    return OneShotObjectDetector(state)

class OneShotObjectDetector(_CustomModel):
    """
    An trained model that is ready to use for classification, exported to
    Core ML, or for feature extraction.

    This model should not be constructed directly.
    """
    _PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION = 1

    def __init__(self, state):
        if False:
            i = 10
            return i + 15
        self.__proxy__ = _PythonProxy(state)

    def predict(self, dataset, confidence_threshold=0.25, iou_threshold=0.45, verbose=True):
        if False:
            return 10
        "\n        Predict object instances in an SFrame of images.\n\n        Parameters\n        ----------\n        dataset : SFrame | SArray | turicreate.Image\n            The images on which to perform object detection.\n            If dataset is an SFrame, it must have a column with the same name\n            as the feature column during training. Additional columns are\n            ignored.\n\n        confidence_threshold : float\n            Only return predictions above this level of confidence. The\n            threshold can range from 0 to 1.\n\n        iou_threshold : float\n            Threshold value for non-maximum suppression. Non-maximum suppression\n            prevents multiple bounding boxes appearing over a single object.\n            This threshold, set between 0 and 1, controls how aggressive this\n            suppression is. A value of 1 means no maximum suppression will\n            occur, while a value of 0 will maximally suppress neighboring\n            boxes around a prediction.\n\n        verbose : bool\n            If True, prints prediction progress.\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions. Each element corresponds to\n            an image and contains a list of dictionaries. Each dictionary\n            describes an object instances that was found in the image. If\n            `dataset` is a single image, the return value will be a single\n            prediction.\n\n        See Also\n        --------\n        draw_bounding_boxes\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            # Make predictions\n            >>> pred = model.predict(data)\n            >>> predictions_with_bounding_boxes = tc.one_shot_object_detector.util.draw_bounding_boxes(data['images'], pred)\n            >>> predictions_with_bounding_boxes.explore()\n\n        "
        return self.__proxy__['detector'].predict(dataset=dataset, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold, verbose=verbose)

    def export_coreml(self, filename, include_non_maximum_suppression=True, iou_threshold=None, confidence_threshold=None):
        if False:
            print('Hello World!')
        "\n        Save the model in Core ML format. The Core ML model takes an image of\n        fixed size as input and produces two output arrays: `confidence` and\n        `coordinates`.\n\n        The first one, `confidence` is an `N`-by-`C` array, where `N` is the\n        number of instances predicted and `C` is the number of classes. The\n        number `N` is fixed and will include many low-confidence predictions.\n        The instances are not sorted by confidence, so the first one will\n        generally not have the highest confidence (unlike in `predict`). Also\n        unlike the `predict` function, the instances have not undergone\n        what is called `non-maximum suppression`, which means there could be\n        several instances close in location and size that have all discovered\n        the same object instance. Confidences do not need to sum to 1 over the\n        classes; any remaining probability is implied as confidence there is no\n        object instance present at all at the given coordinates. The classes\n        appear in the array alphabetically sorted.\n\n        The second array `coordinates` is of size `N`-by-4, where the first\n        dimension `N` again represents instances and corresponds to the\n        `confidence` array. The second dimension represents `x`, `y`, `width`,\n        `height`, in that order.  The values are represented in relative\n        coordinates, so (0.5, 0.5) represents the center of the image and (1,\n        1) the bottom right corner. You will need to multiply the relative\n        values with the original image size before you resized it to the fixed\n        input size to get pixel-value coordinates similar to `predict`.\n\n        See Also\n        --------\n        save\n\n        Parameters\n        ----------\n        filename : string\n            The path of the file where we want to save the Core ML model.\n\n        include_non_maximum_suppression : bool\n            Non-maximum suppression is only available in iOS 12+.\n            A boolean parameter to indicate whether the Core ML model should be\n            saved with built-in non-maximum suppression or not.\n            This parameter is set to True by default.\n\n        iou_threshold : float\n            Threshold value for non-maximum suppression. Non-maximum suppression\n            prevents multiple bounding boxes appearing over a single object.\n            This threshold, set between 0 and 1, controls how aggressive this\n            suppression is. A value of 1 means no maximum suppression will\n            occur, while a value of 0 will maximally suppress neighboring\n            boxes around a prediction.\n\n        confidence_threshold : float\n            Only return predictions above this level of confidence. The\n            threshold can range from 0 to 1.\n\n        Examples\n        --------\n        >>> model.export_coreml('one_shot.mlmodel')\n        "
        from turicreate.toolkits import _coreml_utils
        additional_user_defined_metadata = _coreml_utils._get_tc_version_info()
        short_description = _coreml_utils._mlmodel_short_description('Object Detector')
        options = {'include_non_maximum_suppression': include_non_maximum_suppression}
        options['version'] = self._PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION
        if confidence_threshold is not None:
            options['confidence_threshold'] = confidence_threshold
        if iou_threshold is not None:
            options['iou_threshold'] = iou_threshold
        additional_user_defined_metadata = _coreml_utils._get_tc_version_info()
        short_description = _coreml_utils._mlmodel_short_description('One Shot Object Detector')
        self.__proxy__['detector'].__proxy__.export_to_coreml(filename, short_description, additional_user_defined_metadata, options)

    def _get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return self._PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION

    @classmethod
    def _native_name(cls):
        if False:
            return 10
        return 'one_shot_object_detector'

    def _get_native_state(self):
        if False:
            for i in range(10):
                print('nop')
        state = self.__proxy__.get_state()
        state['detector'] = {'detector_model': state['detector'].__proxy__}
        return state

    @classmethod
    def _load_version(cls, state, version):
        if False:
            print('Hello World!')
        assert version == cls._PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION
        state['detector'] = _ObjectDetector._load_version(state['detector'], state['_detector_version'])
        return OneShotObjectDetector(state)

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the OneShotObjectDetector\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        width = 40
        (sections, section_titles) = self._get_summary_struct()
        detector = self.__proxy__['detector']
        out = _tkutl._toolkit_repr_print(detector, sections, section_titles, width=width, class_name='OneShotObjectDetector')
        return out

    def summary(self, output=None):
        if False:
            return 10
        "\n        Print a summary of the model. The summary includes a description of\n        training data, options, hyper-parameters, and statistics measured\n        during model creation.\n\n        Parameters\n        ----------\n        output : str, None\n            The type of summary to return.\n\n            - None or 'stdout' : print directly to stdout.\n\n            - 'str' : string of summary\n\n            - 'dict' : a dict with 'sections' and 'section_titles' ordered\n              lists. The entries in the 'sections' list are tuples of the form\n              ('label', 'value').\n\n        Examples\n        --------\n        >>> m.summary()\n        "
        from turicreate.toolkits._internal_utils import _toolkit_serialize_summary_struct
        if output is None or output == 'stdout':
            pass
        elif output == 'str':
            return self.__repr__()
        elif output == 'dict':
            return _toolkit_serialize_summary_struct(self.__proxy__['detector'], *self._get_summary_struct())
        try:
            print(self.__repr__())
        except:
            return self.__class__.__name__

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Number of classes', 'num_classes'), ('Input image shape', 'input_image_shape')]
        data_fields = [('Number of synthetically generated examples', 'num_examples'), ('Number of synthetically generated bounding boxes', 'num_bounding_boxes')]
        training_fields = [('Training time', '_training_time_as_string'), ('Training iterations', 'training_iterations'), ('Training epochs', 'training_epochs'), ('Final loss (specific to model)', 'training_loss')]
        section_titles = ['Model summary', 'Synthetic data summary', 'Training summary']
        return ([model_fields, data_fields, training_fields], section_titles)