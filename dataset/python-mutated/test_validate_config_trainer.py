import pytest
from ludwig.config_validation.validation import check_schema
from ludwig.constants import TRAINER
from ludwig.error import ConfigValidationError
from ludwig.schema.optimizers import optimizer_registry
from ludwig.schema.trainer import ECDTrainerConfig
from tests.integration_tests.utils import binary_feature, category_feature, number_feature

def test_config_trainer_empty_null_and_default():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet'}, TRAINER: {}}
    check_schema(config)
    config[TRAINER] = None
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER] = ECDTrainerConfig.Schema().dump({})
    check_schema(config)

def test_config_trainer_bad_optimizer():
    if False:
        while True:
            i = 10
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet'}, TRAINER: {}}
    check_schema(config)
    config[TRAINER]['optimizer'] = None
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    assert ECDTrainerConfig.Schema().load({}).optimizer is not None
    for key in optimizer_registry.keys():
        config[TRAINER]['optimizer'] = {'type': key}
        check_schema(config)
    config[TRAINER]['optimizer'] = {'type': 0}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['optimizer'] = {'type': 'invalid'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_optimizer_property_validation():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet'}, TRAINER: {}}
    check_schema(config)
    config[TRAINER]['optimizer'] = {'type': 'rmsprop'}
    check_schema(config)
    config[TRAINER]['optimizer']['momentum'] = 'invalid'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['optimizer']['momentum'] = 10
    config[TRAINER]['optimizer']['extra_key'] = 'invalid'
    check_schema(config)
    assert not hasattr(ECDTrainerConfig.Schema().load(config[TRAINER]).optimizer, 'extra_key')
    config[TRAINER]['optimizer'] = {'type': 'rmsprop', 'eps': -1}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['optimizer'] = {'type': 'adam', 'betas': (0.1, 0.1)}
    check_schema(config)

def test_clipper_property_validation():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet'}, TRAINER: {}}
    check_schema(config)
    config[TRAINER]['gradient_clipping'] = None
    check_schema(config)
    config[TRAINER]['gradient_clipping'] = {}
    check_schema(config)
    assert ECDTrainerConfig.Schema().load(config[TRAINER]).gradient_clipping == ECDTrainerConfig.Schema().load({}).gradient_clipping
    config[TRAINER]['gradient_clipping'] = 0
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['gradient_clipping'] = 'invalid'
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['gradient_clipping'] = {'clipglobalnorm': None}
    check_schema(config)
    config[TRAINER]['gradient_clipping'] = {'clipglobalnorm': 1}
    check_schema(config)
    config[TRAINER]['gradient_clipping'] = {'clipglobalnorm': 'invalid'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]['gradient_clipping'] = {'clipnorm': 1}
    config[TRAINER]['gradient_clipping']['extra_key'] = 'invalid'
    assert not hasattr(ECDTrainerConfig.Schema().load(config[TRAINER]).gradient_clipping, 'extra_key')