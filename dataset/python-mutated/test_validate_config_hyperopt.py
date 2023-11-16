from itertools import repeat
from unittest.mock import patch
import pytest
import ludwig.schema.hyperopt.parameter
import ludwig.schema.hyperopt.scheduler
import ludwig.schema.hyperopt.search_algorithm
from ludwig.constants import EXECUTOR, HYPEROPT, INPUT_FEATURES, OUTPUT_FEATURES, PARAMETERS, SCHEDULER, SEARCH_ALG, TYPE
from ludwig.error import ConfigValidationError
from ludwig.schema.hyperopt import utils
from ludwig.schema.model_types.base import ModelConfig
from tests.integration_tests.utils import binary_feature, text_feature

@pytest.mark.parametrize('dependencies,raises_exception', [([], False), ([('ludwig', 'ludwig')], False), ([('ludwig', 'ludwig'), ('marshmallow', 'marshmallow')], False), ([('fake_dependency', 'fake_dependency')], True), ([('ludwig', 'ludwig'), ('fake_dependency', 'fake_dependency')], True)])
def test_check_scheduler_dependencies_installed(dependencies, raises_exception):
    if False:
        print('Hello World!')
    config = {INPUT_FEATURES: [text_feature()], OUTPUT_FEATURES: [binary_feature()], HYPEROPT: {PARAMETERS: {'trainer.learning_rate': {'space': 'choice', 'categories': [0.0001, 0.001, 0.01, 0.1]}}, EXECUTOR: {SCHEDULER: {TYPE: 'fifo'}}}}
    with patch('ludwig.schema.hyperopt.utils.get_scheduler_dependencies', return_value=dependencies):
        if raises_exception:
            with pytest.raises(ConfigValidationError):
                ModelConfig.from_dict(config)
        else:
            ModelConfig.from_dict(config)

@pytest.mark.parametrize('dependencies,raises_exception', [([], False), ([('ludwig', 'ludwig')], False), ([('ludwig', 'ludwig'), ('marshmallow', 'marshmallow')], False), ([('fake_dependency', 'fake_dependency')], True), ([('ludwig', 'ludwig'), ('fake_dependency', 'fake_dependency')], True)])
def test_check_search_algorithm_dependencies_installed(dependencies, raises_exception):
    if False:
        return 10
    config = {INPUT_FEATURES: [text_feature()], OUTPUT_FEATURES: [binary_feature()], HYPEROPT: {PARAMETERS: {'trainer.learning_rate': {'space': 'choice', 'categories': [0.0001, 0.001, 0.01, 0.1]}}, SEARCH_ALG: {TYPE: 'random'}}}
    with patch('ludwig.schema.hyperopt.utils.get_search_algorithm_dependencies', return_value=dependencies):
        if raises_exception:
            with pytest.raises(ConfigValidationError):
                ModelConfig.from_dict(config)
        else:
            ModelConfig.from_dict(config)

@pytest.mark.parametrize('space,raises_exception', list(zip(utils.parameter_config_registry.keys(), repeat(False, len(utils.parameter_config_registry)))) + [('fake_space', True)])
def test_parameter_type_check(space, raises_exception):
    if False:
        for i in range(10):
            print('nop')
    'Test that the parameter type is a valid hyperparameter search space.\n\n    This should only be valid until the search space schema is updated to validate spaces as config objects rather than\n    dicts. That update is non-trivial, so to hold over until it is ready we cast the dicts to the corresponding\n    parameter objects and validate as an aux check. The test covers every valid space and one invalid space.\n    '
    config = {INPUT_FEATURES: [text_feature()], OUTPUT_FEATURES: [binary_feature()], HYPEROPT: {SEARCH_ALG: {TYPE: 'random'}, PARAMETERS: {'trainer.learning_rate': {'space': space}}}}
    if not raises_exception:
        ModelConfig.from_dict(config)
    else:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)

@pytest.mark.parametrize('referenced_parameter,raises_exception', [('trainer.learning_rate', False), ('in_feature.encoder.num_fc_layers', False), ('out_feature.decoder.num_fc_layers', False), ('', True), (' ', True), ('foo.bar', True), ('trainer.bar', True), ('foo.learning_rate', True), ('in_feature.encoder.bar', True), ('in_feature.foo.num_fc_layers', True), ('out_feature.encoder.bar', True), ('out_feature.foo.num_fc_layers', True)])
def test_parameter_key_check(referenced_parameter, raises_exception):
    if False:
        i = 10
        return i + 15
    'Test that references to config parameters are validated correctly.\n\n    Hyperopt parameters reference the config parameters they search with `.` notation to access different subsections,\n    e.g. `trainer.learning_rate`. These are added to the config as arbitrary strings, and an invalid reference should be\n    considered a validation error since we will otherwise search over an unused space or defer the error to train time.\n    '
    config = {INPUT_FEATURES: [text_feature(name='in_feature')], OUTPUT_FEATURES: [binary_feature(name='out_feature')], HYPEROPT: {SEARCH_ALG: {TYPE: 'random'}, PARAMETERS: {referenced_parameter: {'space': 'choice', 'categories': [1, 2, 3, 4]}}}}
    if raises_exception:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)
    else:
        ModelConfig.from_dict(config)

@pytest.mark.parametrize('categories,raises_exception', [([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], False), ([{'foo': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'foo': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'foo': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'foo': {'batch_size': 256}}], True), ([{'combiner': {'bar': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'bar': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bar': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], False), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'bar': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'bar': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'bar': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'batch_size': 256}}], True), ([{'combiner': {'type': 'tabnet', 'bn_virtual_bs': 256}, 'trainer': {'learning_rate': 0.001, 'batch_size': 64}}, {'combiner': {'type': 'concat'}, 'trainer': {'bar': 256}}], True)])
def test_nested_parameter_key_check(categories, raises_exception):
    if False:
        while True:
            i = 10
    'Test that nested parameters are validated correctly.'
    config = {INPUT_FEATURES: [text_feature(name='in_feature')], OUTPUT_FEATURES: [binary_feature(name='out_feature')], HYPEROPT: {SEARCH_ALG: {TYPE: 'random'}, PARAMETERS: {'.': {'space': 'choice', 'categories': categories}}}}
    if raises_exception:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)
    else:
        ModelConfig.from_dict(config)

@pytest.mark.parametrize('config', [{'out_feature.decoder.fc_layers': {'space': 'choice', 'categories': [[{'output_size': 64}, {'output_size': 32}], [{'output_size': 64}], [{'output_size': 32}]]}}])
def test_flat_parameter_edge_cases(config):
    if False:
        return 10
    config = {INPUT_FEATURES: [text_feature(name='in_feature')], OUTPUT_FEATURES: [binary_feature(name='out_feature')], HYPEROPT: {SEARCH_ALG: {TYPE: 'random'}, PARAMETERS: config}}
    ModelConfig.from_dict(config)