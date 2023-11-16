from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits import _coreml_utils
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate._deps.minimal_package import _minimal_package_import_check
from .. import _pre_trained_models
from turicreate.toolkits._model import Model as _Model
from turicreate.toolkits._main import ToolkitError as _ToolkitError
import turicreate as _tc
from .._mps_utils import MpsStyleGraphAPI as _MpsStyleGraphAPI

def _get_mps_st_net(input_image_shape, batch_size, output_size, config, weights={}):
    if False:
        print('Hello World!')
    '\n    Initializes an MpsGraphAPI for style transfer.\n    '
    (c_in, h_in, w_in) = input_image_shape
    c_out = output_size[0]
    h_out = h_in
    w_out = w_in
    network = _MpsStyleGraphAPI(batch_size, c_in, h_in, w_in, c_out, h_out, w_out, weights=weights, config=config)
    return network

def create(style_dataset, content_dataset, style_feature=None, content_feature=None, max_iterations=None, model='resnet-16', verbose=True, batch_size=1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a :class:`StyleTransfer` model.\n\n    Parameters\n    ----------\n    style_dataset: SFrame\n        Input style images. The columns named by the ``style_feature`` parameters will\n        be extracted for training the model.\n\n    content_dataset : SFrame\n        Input content images. The columns named by the ``content_feature`` parameters will\n        be extracted for training the model.\n\n    style_feature: string\n        Name of the column containing the input images in style SFrame.\n        \'None\' (the default) indicates the only image column in the style SFrame\n        should be used as the feature.\n\n    content_feature: string\n        Name of the column containing the input images in content SFrame.\n        \'None\' (the default) indicates the only image column in the content\n        SFrame should be used as the feature.\n\n    max_iterations : int\n        The number of training iterations. If \'None\' (the default), then it will\n        be automatically determined based on the amount of data you provide.\n\n    model : string optional\n        Style transfer model to use:\n\n            - "resnet-16" : Fast and small-sized residual network that uses\n                            VGG-16 as reference network during training.\n\n    batch_size : int, optional\n        If you are getting memory errors, try decreasing this value. If you\n        have a powerful computer, increasing this value may improve training\n        throughput.\n\n    verbose : bool, optional\n        If True, print progress updates and model details.\n\n\n    Returns\n    -------\n    out : StyleTransfer\n        A trained :class:`StyleTransfer` model.\n\n    See Also\n    --------\n    StyleTransfer\n\n    Examples\n    --------\n    .. sourcecode:: python\n\n        # Create datasets\n        >>> content_dataset = turicreate.image_analysis.load_images(\'content_images/\')\n        >>> style_dataset = turicreate.image_analysis.load_images(\'style_images/\')\n\n        # Train a style transfer model\n        >>> model = turicreate.style_transfer.create(content_dataset, style_dataset)\n\n        # Stylize an image on all styles\n        >>> stylized_images = model.stylize(data)\n\n        # Visualize the stylized images\n        >>> stylized_images.explore()\n\n    '
    if not isinstance(style_dataset, _tc.SFrame):
        raise TypeError('"style_dataset" must be of type SFrame.')
    if not isinstance(content_dataset, _tc.SFrame):
        raise TypeError('"content_dataset" must be of type SFrame.')
    if len(style_dataset) == 0:
        raise _ToolkitError('style_dataset SFrame cannot be empty')
    if len(content_dataset) == 0:
        raise _ToolkitError('content_dataset SFrame cannot be empty')
    if batch_size < 1:
        raise _ToolkitError("'batch_size' must be greater than or equal to 1")
    if max_iterations is not None and (not isinstance(max_iterations, int) or max_iterations < 0):
        raise _ToolkitError("'max_iterations' must be an integer greater than or equal to 0")
    if style_feature is None:
        style_feature = _tkutl._find_only_image_column(style_dataset)
    if content_feature is None:
        content_feature = _tkutl._find_only_image_column(content_dataset)
    if verbose:
        print("Using '{}' in style_dataset as feature column and using '{}' in content_dataset as feature column".format(style_feature, content_feature))
    _raise_error_if_not_training_sframe(style_dataset, style_feature)
    _raise_error_if_not_training_sframe(content_dataset, content_feature)
    _tkutl._handle_missing_values(style_dataset, style_feature, 'style_dataset')
    _tkutl._handle_missing_values(content_dataset, content_feature, 'content_dataset')
    params = {'batch_size': batch_size, 'vgg16_content_loss_layer': 2, 'lr': 0.001, 'content_loss_mult': 1.0, 'style_loss_mult': [0.0001, 0.0001, 0.0001, 0.0001], 'finetune_all_params': True, 'pretrained_weights': False, 'print_loss_breakdown': False, 'input_shape': (256, 256), 'training_content_loader_type': 'stretch', 'use_augmentation': False, 'sequential_image_processing': False, 'aug_resize': 0, 'aug_min_object_covered': 0, 'aug_rand_crop': 0.9, 'aug_rand_pad': 0.9, 'aug_rand_gray': 0.0, 'aug_aspect_ratio': 1.25, 'aug_hue': 0.05, 'aug_brightness': 0.05, 'aug_saturation': 0.05, 'aug_contrast': 0.05, 'aug_horizontal_flip': True, 'aug_area_range': (0.05, 1.5), 'aug_pca_noise': 0.0, 'aug_max_attempts': 20, 'aug_inter_method': 2, 'checkpoint': False, 'checkpoint_prefix': 'style_transfer', 'checkpoint_increment': 1000}
    if '_advanced_parameters' in kwargs:
        new_keys = set(kwargs['_advanced_parameters'].keys())
        set_keys = set(params.keys())
        unsupported = new_keys - set_keys
        if unsupported:
            raise _ToolkitError('Unknown advanced parameters: {}'.format(unsupported))
        params.update(kwargs['_advanced_parameters'])
    name = 'style_transfer'
    import turicreate as _turicreate
    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
    model = _turicreate.extensions.style_transfer()
    pretrained_resnet_model = _pre_trained_models.STYLE_TRANSFER_BASE_MODELS['resnet-16']()
    pretrained_vgg16_model = _pre_trained_models.STYLE_TRANSFER_BASE_MODELS['Vgg16']()
    options = {}
    options['image_height'] = params['input_shape'][0]
    options['image_width'] = params['input_shape'][1]
    options['content_feature'] = content_feature
    options['style_feature'] = style_feature
    if verbose is not None:
        options['verbose'] = verbose
    else:
        options['verbose'] = False
    if batch_size is not None:
        options['batch_size'] = batch_size
    if max_iterations is not None:
        options['max_iterations'] = max_iterations
    options['num_styles'] = len(style_dataset)
    options['resnet_mlmodel_path'] = pretrained_resnet_model.get_model_path('coreml')
    options['vgg_mlmodel_path'] = pretrained_vgg16_model.get_model_path('coreml')
    options['pretrained_weights'] = params['pretrained_weights']
    model.train(style_dataset[style_feature], content_dataset[content_feature], options)
    return StyleTransfer(model_proxy=model, name=name)

def _raise_error_if_not_training_sframe(dataset, context_column):
    if False:
        i = 10
        return i + 15
    _raise_error_if_not_sframe(dataset, 'datset')
    if context_column not in dataset.column_names():
        raise _ToolkitError("Context Image column '%s' does not exist" % context_column)
    if dataset[context_column].dtype != _tc.Image:
        raise _ToolkitError('Context Image column must contain images')

class StyleTransfer(_Model):
    """
    A trained model using C++ implementation that is ready to use for classification or export to
    CoreML.

    This model should not be constructed directly.
    """
    _CPP_STYLE_TRANSFER_VERSION = 1

    def __init__(self, model_proxy=None, name=None):
        if False:
            return 10
        self.__proxy__ = model_proxy
        self.__name__ = name

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        return 'style_transfer'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the StyleTransfer.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        width = 40
        (sections, section_titles) = self._get_summary_struct()
        out = _tkutl._toolkit_repr_print(self, sections, section_titles, width=width)
        return out

    def _get_version(self):
        if False:
            return 10
        return self._CPP_STYLE_TRANSFER_VERSION

    def export_coreml(self, filename, image_shape=(256, 256), include_flexible_shape=True):
        if False:
            i = 10
            return i + 15
        "\n        Save the model in Core ML format. The Core ML model takes an image of\n        fixed size, and a style index inputs and produces an output\n        of an image of fixed size\n\n        Parameters\n        ----------\n        path : string\n            A string to the path for saving the Core ML model.\n\n        image_shape: tuple\n            A tuple (defaults to (256, 256)) will bind the coreml model to a fixed shape.\n\n        include_flexible_shape: bool\n            Allows the size of the input image to be flexible. Any input image were the\n            height and width are at least 64 will be accepted by the Core ML Model.\n\n        See Also\n        --------\n        save\n\n        Examples\n        --------\n        >>> model.export_coreml('StyleTransfer.mlmodel')\n        "
        options = {}
        options['image_width'] = image_shape[1]
        options['image_height'] = image_shape[0]
        options['include_flexible_shape'] = include_flexible_shape
        additional_user_defined_metadata = _coreml_utils._get_tc_version_info()
        short_description = _coreml_utils._mlmodel_short_description('Style Transfer')
        self.__proxy__.export_to_coreml(filename, short_description, additional_user_defined_metadata, options)

    def stylize(self, images, style=None, verbose=True, max_size=800, batch_size=4):
        if False:
            return 10
        '\n        Stylize an SFrame of Images given a style index or a list of\n        styles.\n\n        Parameters\n        ----------\n        images : SFrame | SArray | turicreate.Image\n            A dataset that has the same content image column that was used\n            during training.\n\n        style : None | int | list\n            The selected style or list of styles to use on the ``images``. If\n            `None`, all styles will be applied to each image in ``images``.\n\n        verbose : bool, optional\n            If True, print progress updates.\n\n        max_size : int or tuple\n            Max input image size that will not get resized during stylization.\n\n            Images with a side larger than this value, will be scaled down, due\n            to time and memory constraints. If tuple, interpreted as (max\n            width, max height). Without resizing, larger input images take more\n            time to stylize.  Resizing can effect the quality of the final\n            stylized image.\n\n        batch_size : int, optional\n            If you are getting memory errors, try decreasing this value. If you\n            have a powerful computer, increasing this value may improve\n            performance.\n\n        Returns\n        -------\n        out : SFrame or SArray or turicreate.Image\n            If ``style`` is a list, an SFrame is always returned. If ``style``\n            is a single integer, the output type will match the input type\n            (Image, SArray, or SFrame).\n\n        See Also\n        --------\n        create\n\n        Examples\n        --------\n        >>> image = tc.Image("/path/to/image.jpg")\n        >>> stylized_images = model.stylize(image, style=[0, 1])\n        Data:\n        +--------+-------+------------------------+\n        | row_id | style |     stylized_image     |\n        +--------+-------+------------------------+\n        |   0    |   0   | Height: 256 Width: 256 |\n        |   0    |   1   | Height: 256 Width: 256 |\n        +--------+-------+------------------------+\n        [2 rows x 3 columns]\n\n        >>> images = tc.image_analysis.load_images(\'/path/to/images\')\n        >>> stylized_images = model.stylize(images)\n        Data:\n        +--------+-------+------------------------+\n        | row_id | style |     stylized_image     |\n        +--------+-------+------------------------+\n        |   0    |   0   | Height: 256 Width: 256 |\n        |   0    |   1   | Height: 256 Width: 256 |\n        |   0    |   2   | Height: 256 Width: 256 |\n        |   0    |   3   | Height: 256 Width: 256 |\n        |   1    |   0   | Height: 640 Width: 648 |\n        |   1    |   1   | Height: 640 Width: 648 |\n        |   1    |   2   | Height: 640 Width: 648 |\n        |   1    |   3   | Height: 640 Width: 648 |\n        +--------+-------+------------------------+\n        [8 rows x 3 columns]\n        '
        if not isinstance(images, (_tc.SFrame, _tc.SArray, _tc.Image)):
            raise TypeError('"image" parameter must be of type SFrame, SArray or turicreate.Image.')
        if isinstance(images, (_tc.SFrame, _tc.SArray)) and len(images) == 0:
            raise _ToolkitError('"image" parameter cannot be empty')
        if style is not None and (not isinstance(style, (int, list))):
            raise TypeError('"style" must parameter must be a None, int or a list')
        if not isinstance(max_size, int):
            raise TypeError('"max_size" must parameter must be an int')
        if max_size < 1:
            raise _ToolkitError("'max_size' must be greater than or equal to 1")
        if not isinstance(batch_size, int):
            raise TypeError('"batch_size" must parameter must be an int')
        if batch_size < 1:
            raise _ToolkitError("'batch_size' must be greater than or equal to 1")
        options = {}
        options['style_idx'] = style
        options['verbose'] = verbose
        options['max_size'] = max_size
        options['batch_size'] = batch_size
        if isinstance(style, list) or style is None:
            if isinstance(images, _tc.SFrame):
                image_feature = _tkutl._find_only_image_column(images)
                stylized_images = self.__proxy__.predict(images[image_feature], options)
                stylized_images = stylized_images.rename({'stylized_image': 'stylized_' + str(image_feature)})
                return stylized_images
            return self.__proxy__.predict(images, options)
        elif isinstance(images, _tc.SFrame):
            if len(images) == 0:
                raise _ToolkitError('SFrame cannot be empty')
            image_feature = _tkutl._find_only_image_column(images)
            stylized_images = self.__proxy__.predict(images[image_feature], options)
            stylized_images = stylized_images.rename({'stylized_image': 'stylized_' + str(image_feature)})
            return stylized_images
        elif isinstance(images, _tc.Image):
            stylized_images = self.__proxy__.predict(images, options)
            return stylized_images['stylized_image'][0]
        elif isinstance(images, _tc.SArray):
            stylized_images = self.__proxy__.predict(images, options)
            return stylized_images['stylized_image']

    def get_styles(self, style=None):
        if False:
            print('Hello World!')
        '\n        Returns SFrame of style images used for training the model\n\n        Parameters\n        ----------\n        style: int or list, optional\n            The selected style or list of styles to return. If `None`, all\n            styles will be returned\n\n        See Also\n        --------\n        stylize\n\n        Examples\n        --------\n        >>>  model.get_styles()\n        Columns:\n            style   int\n            image   Image\n\n        Rows: 4\n\n        Data:\n        +-------+--------------------------+\n        | style |          image           |\n        +-------+--------------------------+\n        |  0    |  Height: 642 Width: 642  |\n        |  1    |  Height: 642 Width: 642  |\n        |  2    |  Height: 642 Width: 642  |\n        |  3    |  Height: 642 Width: 642  |\n        +-------+--------------------------+\n        '
        return self.__proxy__.get_styles(style)

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Model', 'model'), ('Number of unique styles', 'num_styles')]
        training_fields = [('Training time', '_training_time_as_string'), ('Training epochs', 'training_epochs'), ('Training iterations', 'training_iterations'), ('Number of style images', 'num_styles'), ('Number of content images', 'num_content_images'), ('Final loss', 'training_loss')]
        section_titles = ['Schema', 'Training summary']
        return ([model_fields, training_fields], section_titles)