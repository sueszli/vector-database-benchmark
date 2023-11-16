from copy import deepcopy
from typing import Dict
import pytest
import torch
from ludwig.constants import BFILL, CROP_OR_PAD, ENCODER, ENCODER_OUTPUT, INTERPOLATE, TYPE
from ludwig.features.image_feature import _ImagePreprocessing, ImageInputFeature
from ludwig.schema.features.image_feature import ImageInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.torch_utils import get_torch_device
BATCH_SIZE = 2
DEVICE = get_torch_device()

@pytest.fixture(scope='module')
def image_config():
    if False:
        return 10
    return {'name': 'image_column_name', 'type': 'image', 'tied': None, 'encoder': {'type': 'stacked_cnn', 'conv_layers': None, 'num_conv_layers': None, 'filter_size': 3, 'num_filters': 256, 'strides': (1, 1), 'padding': 'valid', 'dilation_rate': (1, 1), 'conv_use_bias': True, 'conv_weights_initializer': 'xavier_uniform', 'conv_bias_initializer': 'zeros', 'conv_norm': None, 'conv_norm_params': None, 'conv_activation': 'relu', 'conv_dropout': 0, 'pool_function': 'max', 'pool_size': (2, 2), 'pool_strides': None, 'fc_layers': None, 'num_fc_layers': 1, 'output_size': 16, 'fc_use_bias': True, 'fc_weights_initializer': 'xavier_uniform', 'fc_bias_initializer': 'zeros', 'fc_norm': None, 'fc_norm_params': None, 'fc_activation': 'relu', 'fc_dropout': 0}, 'preprocessing': {'height': 28, 'width': 28, 'num_channels': 1, 'scaling': 'pixel_normalization'}}

@pytest.mark.parametrize('encoder, height, width, num_channels', [('stacked_cnn', 28, 28, 3), ('stacked_cnn', 28, 28, 1), ('mlp_mixer', 32, 32, 3)])
def test_image_input_feature(image_config: Dict, encoder: str, height: int, width: int, num_channels: int) -> None:
    if False:
        while True:
            i = 10
    image_def = deepcopy(image_config)
    image_def[ENCODER][TYPE] = encoder
    image_def[ENCODER]['height'] = height
    image_def[ENCODER]['width'] = width
    image_def[ENCODER]['num_channels'] = num_channels
    defaults = ImageInputFeatureConfig(name='foo').to_dict()
    set_def = merge_dict(defaults, image_def)
    (image_config, _) = load_config_with_kwargs(ImageInputFeatureConfig, set_def)
    input_feature_obj = ImageInputFeature(image_config).to(DEVICE)
    input_tensor = torch.rand(size=(BATCH_SIZE, num_channels, height, width), dtype=torch.float32).to(DEVICE)
    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, *input_feature_obj.output_shape)

def test_image_preproc_module_bad_num_channels():
    if False:
        return 10
    metadata = {'preprocessing': {'missing_value_strategy': BFILL, 'in_memory': True, 'resize_method': 'interpolate', 'scaling': 'pixel_normalization', 'num_processes': 1, 'infer_image_num_channels': True, 'infer_image_dimensions': True, 'infer_image_max_height': 256, 'infer_image_max_width': 256, 'infer_image_sample_size': 100, 'height': 12, 'width': 12, 'num_channels': 2}, 'reshape': (2, 12, 12)}
    module = _ImagePreprocessing(metadata)
    with pytest.raises(ValueError):
        module(torch.rand(2, 3, 10, 10))

@pytest.mark.parametrize('resize_method', [INTERPOLATE, CROP_OR_PAD])
@pytest.mark.parametrize(['num_channels', 'num_channels_expected'], [(1, 3), (3, 1)])
def test_image_preproc_module_list_of_tensors(resize_method, num_channels, num_channels_expected):
    if False:
        for i in range(10):
            print('nop')
    metadata = {'preprocessing': {'missing_value_strategy': BFILL, 'in_memory': True, 'resize_method': resize_method, 'scaling': 'pixel_normalization', 'num_processes': 1, 'infer_image_num_channels': True, 'infer_image_dimensions': True, 'infer_image_max_height': 256, 'infer_image_max_width': 256, 'infer_image_sample_size': 100, 'height': 12, 'width': 12, 'num_channels': num_channels_expected}, 'reshape': (num_channels_expected, 12, 12)}
    module = _ImagePreprocessing(metadata)
    res = module([torch.rand(num_channels, 25, 25), torch.rand(num_channels, 10, 10)])
    assert res.shape == torch.Size((2, num_channels_expected, 12, 12))

@pytest.mark.parametrize('resize_method', [INTERPOLATE, CROP_OR_PAD])
@pytest.mark.parametrize(['num_channels', 'num_channels_expected'], [(1, 3), (3, 1)])
def test_image_preproc_module_tensor(resize_method, num_channels, num_channels_expected):
    if False:
        i = 10
        return i + 15
    metadata = {'preprocessing': {'missing_value_strategy': BFILL, 'in_memory': True, 'resize_method': resize_method, 'scaling': 'pixel_normalization', 'num_processes': 1, 'infer_image_num_channels': True, 'infer_image_dimensions': True, 'infer_image_max_height': 256, 'infer_image_max_width': 256, 'infer_image_sample_size': 100, 'height': 12, 'width': 12, 'num_channels': num_channels_expected}, 'reshape': (num_channels_expected, 12, 12)}
    module = _ImagePreprocessing(metadata)
    res = module(torch.rand(2, num_channels, 10, 10))
    assert res.shape == torch.Size((2, num_channels_expected, 12, 12))