import pytest
from ludwig.config_validation.validation import check_schema, get_schema
from ludwig.constants import MODEL_ECD, TRAINER
from ludwig.error import ConfigValidationError
from tests.integration_tests.utils import binary_feature, category_feature, number_feature

def test_combiner_schema_is_not_empty_for_ECD():
    if False:
        for i in range(10):
            print('nop')
    assert len(get_schema(MODEL_ECD)['properties']['combiner']['allOf']) > 0

@pytest.mark.parametrize('eval_batch_size', [500000, None])
def test_config_tabnet(eval_batch_size):
    if False:
        print('Hello World!')
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet', 'size': 24, 'output_size': 26, 'sparsity': 1e-06, 'bn_virtual_divider': 32, 'bn_momentum': 0.4, 'num_steps': 5, 'relaxation_factor': 1.5, 'use_keras_batch_norm': False, 'bn_virtual_bs': 512}, TRAINER: {'batch_size': 16384, 'eval_batch_size': eval_batch_size, 'epochs': 1000, 'early_stop': 20, 'learning_rate': 0.02, 'optimizer': {'type': 'adam'}, 'learning_rate_scheduler': {'decay': 'linear', 'decay_steps': 20000, 'decay_rate': 0.9, 'staircase': True}, 'regularization_lambda': 1, 'regularization_type': 'l2'}}
    check_schema(config)

def test_config_bad_combiner():
    if False:
        print('Hello World!')
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet'}}
    check_schema(config)
    del config['combiner']['type']
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['type'] = 'fake'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner'] = [{'type': 'tabnet'}]
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner'] = {'type': 'tabtransformer', 'num_layers': 10, 'dropout': False}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner'] = {'type': 'transformer', 'dropout': -1}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_config_bad_combiner_types_enums():
    if False:
        return 10
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'concat', 'weights_initializer': 'zeros'}}
    check_schema(config)
    config['combiner']['weights_initializer'] = {'test': 'fail'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['weights_initializer'] = 'fail'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['weights_initializer'] = {}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['weights_initializer'] = {'type': 'fail'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['weights_initializer'] = {'type': 'normal', 'stddev': 0}
    check_schema(config)
    del config['combiner']['weights_initializer']
    config['combiner']['bias_initializer'] = 'kaiming_uniform'
    check_schema(config)
    config['combiner']['bias_initializer'] = 'fail'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['bias_initializer'] = {}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['bias_initializer'] = {'type': 'fail'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config['combiner']['bias_initializer'] = {'type': 'zeros', 'stddev': 0}
    check_schema(config)
    del config['combiner']['bias_initializer']
    config['combiner']['norm'] = 'batch'
    check_schema(config)
    config['combiner']['norm'] = 'fail'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    del config['combiner']['norm']
    config['combiner']['activation'] = 'relu'
    check_schema(config)
    config['combiner']['activation'] = 123
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    del config['combiner']['activation']
    config2 = {**config}
    config2['combiner']['type'] = 'tabtransformer'
    config2['combiner']['reduce_output'] = 'sum'
    check_schema(config)
    config2['combiner']['reduce_output'] = 'fail'
    with pytest.raises(ConfigValidationError):
        check_schema(config2)
    config2['combiner']['reduce_output'] = None
    check_schema(config2)