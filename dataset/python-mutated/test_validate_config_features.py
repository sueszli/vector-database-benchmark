import pytest
from ludwig.config_validation.validation import check_schema
from ludwig.error import ConfigValidationError
from tests.integration_tests.utils import binary_feature, category_feature, number_feature, text_feature

def test_config_input_output_features():
    if False:
        while True:
            i = 10
    config = {'input_features': [category_feature(encoder={'type': 'dense'}), number_feature(encoder={'type': 'passthrough'})], 'output_features': [binary_feature(decoder={'type': 'regressor'})]}
    check_schema(config)

def test_incorrect_input_features_config():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [category_feature(preprocessing={'normalization': 'zscore'})], 'output_features': [binary_feature()]}
    config = {'input_features': [text_feature(preprocessing={'padding_symbol': 0})], 'output_features': [binary_feature()]}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config = {'input_features': [binary_feature()], 'output_features': [binary_feature()]}
    del config['input_features'][0]['type']
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_incorrect_output_features_config():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [number_feature()], 'output_features': [binary_feature(decoder='classifier')]}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_too_few_features_config():
    if False:
        for i in range(10):
            print('nop')
    ifeatures = [number_feature()]
    ofeatures = [binary_feature()]
    check_schema({'input_features': ifeatures, 'output_features': ofeatures})
    with pytest.raises(ConfigValidationError):
        check_schema({'input_features': [], 'output_features': ofeatures})
    with pytest.raises(ConfigValidationError):
        check_schema({'input_features': ifeatures, 'output_features': []})

def test_too_many_features_config():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ConfigValidationError):
        check_schema({'input_features': [number_feature()], 'output_features': [binary_feature(), number_feature()], 'model_type': 'gbm'})
    check_schema({'input_features': [number_feature()], 'output_features': [binary_feature(), number_feature()], 'model_type': 'ecd'})