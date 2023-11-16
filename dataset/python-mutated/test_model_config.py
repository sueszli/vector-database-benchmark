import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union
import pytest
import yaml
from ludwig.constants import ACTIVE, BASE_MODEL, CLIP, COLUMN, COMBINER, DECODER, DEFAULT_VALIDATION_METRIC, DEFAULTS, DEPENDENCIES, ENCODER, HYPEROPT, INPUT_FEATURES, INPUT_SIZE, LOSS, MODEL_ECD, MODEL_GBM, MODEL_LLM, MODEL_TYPE, NAME, NUM_CLASSES, OPTIMIZER, OUTPUT_FEATURES, PREPROCESSING, PROC_COLUMN, REDUCE_DEPENDENCIES, REDUCE_INPUT, TIED, TRAINER, TYPE
from ludwig.error import ConfigValidationError
from ludwig.schema.decoders.base import ClassifierConfig
from ludwig.schema.encoders.text_encoders import BERTConfig
from ludwig.schema.features.augmentation.image import RandomBlurConfig, RandomRotateConfig
from ludwig.schema.features.image_feature import AUGMENTATION_DEFAULT_OPERATIONS
from ludwig.schema.features.number_feature import NumberOutputFeatureConfig
from ludwig.schema.features.text_feature import TextOutputFeatureConfig
from ludwig.schema.llms.quantization import QuantizationConfig
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.utils import BaseMarshmallowConfig, convert_submodules
config_sections = {INPUT_FEATURES, OUTPUT_FEATURES, PREPROCESSING, TRAINER, COMBINER, DEFAULTS, HYPEROPT}

def test_config_object():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text', 'preprocessing': {'missing_value_strategy': 'drop_row'}, 'encoder': {'type': 'rnn', 'bidirectional': True, 'representation': 'dense', 'num_layers': 2}}, {'name': 'image_feature_1', 'type': 'image', 'preprocessing': {'height': 32, 'width': 32, 'num_channels': 4}, 'encoder': {'type': 'stacked_cnn', 'num_channels': 4, 'dropout': 0.1}}], 'output_features': [{'name': 'category_feature', 'type': 'category', 'top_k': 3, 'preprocessing': {'missing_value_strategy': 'bfill'}, 'decoder': {'type': 'classifier', 'num_classes': 10, 'use_bias': False}}], 'combiner': {'type': 'concat', 'output_size': 512, 'weights_initializer': 'xavier_uniform', 'dropout': 0.2}, 'trainer': {'epochs': 50, 'batch_size': 'auto', 'optimizer': {'type': 'adam', 'betas': [0.8, 0.999], 'eps': 5e-09}}}
    config_object = ModelConfig.from_dict(config)
    assert config_object.input_features.text_feature.encoder.type == 'rnn'
    assert config_object.input_features.text_feature.encoder.num_layers == 2
    assert config_object.input_features.text_feature.preprocessing.missing_value_strategy == 'drop_row'
    assert config_object.defaults.text.encoder.type != 'rnn'
    assert config_object.defaults.text.preprocessing.missing_value_strategy != 'drop_row'
    assert config_object.output_features.category_feature.decoder.num_classes == 10
    assert config_object.output_features.category_feature.top_k == 3
    assert config_object.combiner.output_size == 512
    assert config_object.combiner.weights_initializer == 'xavier_uniform'
    assert config_object.combiner.fc_layers is None
    assert config_object.trainer.epochs == 50
    assert config_object.trainer.batch_size == 'auto'
    assert config_object.trainer.optimizer.type == 'adam'
    assert config_object.trainer.optimizer.betas[0] == 0.8
    assert config_object.trainer.optimizer.betas[1] == 0.999
    assert config_object.trainer.optimizer.eps == 5e-09

def test_config_object_defaults():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'number_feature', 'type': 'number'}, {'name': 'text_feature_1', 'type': 'text', 'encoder': {'type': 'rnn', 'activation': 'sigmoid'}}, {'name': 'text_feature_2', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}], 'defaults': {'number': {'preprocessing': {'missing_value_strategy': 'drop_row'}, 'encoder': {'type': 'dense'}}, 'text': {'preprocessing': {'missing_value_strategy': 'drop_row'}, 'encoder': {'type': 'stacked_parallel_cnn', 'activation': 'tanh'}}}}
    config_object = ModelConfig.from_dict(config)
    assert config_object.input_features.number_feature.preprocessing.missing_value_strategy == 'drop_row'
    assert config_object.input_features.number_feature.encoder.type == 'dense'
    assert config_object.input_features.text_feature_1.encoder.type == 'rnn'
    assert config_object.input_features.text_feature_1.encoder.activation == 'sigmoid'
    assert config_object.input_features.text_feature_1.preprocessing.missing_value_strategy == 'drop_row'
    assert config_object.input_features.text_feature_2.encoder.type == 'stacked_parallel_cnn'
    assert config_object.input_features.text_feature_2.encoder.activation == 'tanh'
    assert config_object.input_features.text_feature_2.preprocessing.missing_value_strategy == 'drop_row'

def test_config_object_to_config_dict():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'number_feature', 'type': 'number'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_object = ModelConfig.from_dict(config)
    config_dict = config_object.to_dict()
    for section in config_sections:
        assert section in config_dict
    assert len(config_dict[DEFAULTS]) == 13
    assert set(config_dict[INPUT_FEATURES][0].keys()) == {NAME, ACTIVE, TYPE, COLUMN, PROC_COLUMN, TIED, PREPROCESSING, ENCODER}
    assert set(config_dict[OUTPUT_FEATURES][0].keys()) == {NAME, ACTIVE, TYPE, COLUMN, PROC_COLUMN, PREPROCESSING, DECODER, LOSS, REDUCE_INPUT, DEPENDENCIES, INPUT_SIZE, CLIP, REDUCE_DEPENDENCIES, NUM_CLASSES, DEFAULT_VALIDATION_METRIC}

def test_update_config_object():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_object = ModelConfig.from_dict(config)
    assert config_object.input_features.text_feature.encoder.type == 'parallel_cnn'
    assert config_object.input_features.text_feature.encoder.max_sequence_length is None
    temp_config = {'input_features': [{'name': 'text_feature', 'type': 'text', 'encoder': {'type': 'parallel_cnn', 'max_sequence_length': 10}}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_object = ModelConfig.from_dict(temp_config)
    assert config_object.input_features.text_feature.encoder.max_sequence_length == 10

@pytest.mark.parametrize('model_type', [MODEL_ECD, MODEL_GBM])
def test_config_object_validation_parameters_defaults(model_type: str):
    if False:
        while True:
            i = 10
    config = {'input_features': [{'name': 'category_feature', 'type': 'category'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}], 'model_type': model_type}
    config_object = ModelConfig.from_dict(config)
    assert config_object.trainer.validation_field == 'number_output_feature'
    assert config_object.trainer.validation_metric == NumberOutputFeatureConfig.default_validation_metric

def test_config_object_validation_parameters_multiple_output_features():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'text_output_feature', 'type': 'text'}, {'name': 'number_output_feature', 'type': 'number'}]}
    config_object = ModelConfig.from_dict(config)
    assert config_object.trainer.validation_field == 'text_output_feature'
    assert config_object.trainer.validation_metric == TextOutputFeatureConfig.default_validation_metric
    tmp = config['output_features'][0]
    config['output_features'][0] = config['output_features'][1]
    config['output_features'][1] = tmp
    config_object = ModelConfig.from_dict(config)
    assert config_object.trainer.validation_field == 'number_output_feature'
    assert config_object.trainer.validation_metric == NumberOutputFeatureConfig.default_validation_metric

def test_config_object_validation_parameters_explicitly_set_validation_field():
    if False:
        return 10
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'text_output_feature', 'type': 'text'}, {'name': 'number_output_feature', 'type': 'number'}], 'trainer': {'validation_field': 'combined'}}
    config_object = ModelConfig.from_dict(config)
    assert config_object.trainer.validation_field == 'combined'
    assert config_object.trainer.validation_metric == 'loss'

def test_config_object_validation_parameters_explicitly_set_validation_metric():
    if False:
        while True:
            i = 10
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'text_output_feature', 'type': 'text'}, {'name': 'number_output_feature', 'type': 'number'}], 'trainer': {'validation_metric': NumberOutputFeatureConfig.default_validation_metric}}
    config_object = ModelConfig.from_dict(config)
    assert config_object.trainer.validation_field == 'number_output_feature'
    assert config_object.trainer.validation_metric == NumberOutputFeatureConfig.default_validation_metric

def test_config_object_validation_parameters_invalid_metric():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'text_output_feature', 'type': 'text'}], 'trainer': {'validation_metric': NumberOutputFeatureConfig.default_validation_metric}}
    with pytest.raises(Exception):
        ModelConfig.from_dict(config)

def test_config_object_validation_parameters_metric_conflict():
    if False:
        return 10
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature1', 'type': 'number'}, {'name': 'number_output_feature2', 'type': 'number'}], 'trainer': {'validation_metric': NumberOutputFeatureConfig.default_validation_metric}}
    with pytest.raises(Exception):
        ModelConfig.from_dict(config)

def test_constructors_yaml():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text', 'encoder': {'type': 'parallel_cnn', 'max_sequence_length': 10}}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.yaml')
        with open(file_path, 'w') as file:
            yaml.dump(config, file)
        config_obj = ModelConfig.from_yaml(file_path)
    for section in config_sections:
        assert hasattr(config_obj, section)

def test_constructors_dict():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [{'name': 'text_feature', 'type': 'text', 'encoder': {'type': 'parallel_cnn', 'max_sequence_length': 10}}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    for section in config_sections:
        assert hasattr(config_obj, section)

def test_feature_enabling_disabling():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}, {'name': 'category_feature', 'type': 'number'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features.text_feature.active
    assert config_obj.input_features.category_feature.active
    config_obj.input_features.text_feature.disable()
    assert not config_obj.input_features.text_feature.active

def test_sequence_combiner():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}], 'combiner': {'type': 'sequence', 'encoder': {'type': 'rnn'}}}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.combiner.type == 'sequence'
    assert config_obj.combiner.encoder.type == 'rnn'
    assert config_obj.combiner.encoder.cell_type == 'rnn'

@pytest.mark.parametrize('session', [{'sess_id': 0, 'encoder': 'parallel_cnn', 'loss': {'type': 'mean_squared_error'}}, {'sess_id': 1, 'encoder': 'cnnrnn', 'loss': {'type': 'mean_absolute_error'}}, {'sess_id': 2, 'encoder': 'parallel_cnn', 'loss': {'type': 'mean_absolute_error'}}])
def test_shared_state(session):
    if False:
        while True:
            i = 10
    config = {'input_features': [{'name': 'text_feature', 'type': 'text', 'encoder': {'type': session['encoder']}}, {'name': 'text_feature_2', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}, {'name': 'category_feature', 'type': 'category', 'preprocessing': {'missing_value_strategy': 'bfill'}}], 'defaults': {'text': {'encoder': {'type': session['encoder']}}}}
    if session['sess_id'] == 1:
        del config[OUTPUT_FEATURES][1]['preprocessing']
    if session['sess_id'] == 2:
        del config[INPUT_FEATURES][0]['encoder']
        del config[DEFAULTS]
    config_obj = ModelConfig.from_dict(config)
    if session['sess_id'] == 0:
        config_obj.input_features.text_feature.encoder.max_sequence_length = 10
        config_obj.input_features.text_feature.tied = 'text_feature_2'
        assert config_obj.defaults.text.encoder.max_sequence_length is None
        assert config_obj.input_features.text_feature.tied == 'text_feature_2'
    if session['sess_id'] == 1:
        config_obj.output_features.number_output_feature.loss.weight = 2.0
        assert config_obj.output_features.category_feature.preprocessing.missing_value_strategy == 'drop_row'
        assert config_obj.defaults.text.encoder.max_sequence_length is None
        assert config_obj.input_features.text_feature.tied is None
        assert config_obj.output_features.number_output_feature.loss.weight == 2.0
    if session['sess_id'] == 2:
        assert config_obj.input_features.text_feature.encoder.type == 'parallel_cnn'
        assert config_obj.output_features.number_output_feature.loss.weight == 1.0
        assert config_obj.defaults.text.encoder.max_sequence_length is None
        assert config_obj.input_features.text_feature.tied is None

def test_convert_submodules():
    if False:
        while True:
            i = 10
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    trainer = convert_submodules(config_obj.trainer.__dict__)
    input_features = config_obj.input_features.to_list()
    assert not isinstance(trainer[OPTIMIZER], BaseMarshmallowConfig)
    assert not isinstance(input_features[0][PREPROCESSING], BaseMarshmallowConfig)

def test_defaults_mixins():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'text_feature', 'type': 'text'}], 'output_features': [{'name': 'number_output_feature', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.defaults.audio.to_dict().keys() == {ENCODER, PREPROCESSING}
    assert config_obj.defaults.category.to_dict().keys() == {ENCODER, PREPROCESSING, DECODER, LOSS}

def test_initializer_recursion():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'category_B9834', 'type': 'category', 'encoder': {'type': 'dense', 'vocab_size': 2, 'embedding_size': 5}, 'reduce_input': 'sum', 'column': 'category_B9834', 'proc_column': 'category_B9834_mZFLky'}, {'name': 'number_0F633', 'type': 'number', 'encoder': {'type': 'dense', 'norm': 'batch', 'norm_params': {'momentum': 0.2}}}], 'output_features': [{'name': 'binary_52912', 'type': 'binary', 'column': 'binary_52912', 'proc_column': 'binary_52912_mZFLky'}], 'combiner': {'type': 'concat', 'weights_initializer': {'type': 'normal', 'stddev': 0}}}
    config_obj = ModelConfig.from_dict(config)
    assert isinstance(config_obj.combiner.weights_initializer, dict)

def test_number_feature_zscore_preprocessing_default():
    if False:
        print('Hello World!')
    "Tests that the default value for the number feature preprocessing is 'zscore'."
    config = {'input_features': [{'name': 'number_input_feature1', 'type': 'number'}], 'output_features': [{'name': 'number_output_feature1', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features.number_input_feature1.preprocessing.normalization == 'zscore'

@pytest.mark.parametrize('augmentation,expected', [(None, []), (False, []), (True, AUGMENTATION_DEFAULT_OPERATIONS), ([{'type': 'random_blur'}, {'type': 'random_rotate', 'degree': 30}], [RandomBlurConfig(), RandomRotateConfig(degree=30)])])
def test_augmentation_pipeline(augmentation, expected):
    if False:
        while True:
            i = 10
    'Tests that augmentation pipeline is correctly deserialized and serialized between config.'
    config = {'input_features': [{'name': 'input1', 'type': 'image', 'augmentation': augmentation}], 'output_features': [{'name': 'output1', 'type': 'number'}]}
    if augmentation is None:
        del config['input_features'][0]['augmentation']
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].augmentation == expected
    config_dict = config_obj.to_dict()
    assert len(config_dict['input_features'][0]['augmentation']) == len(expected)
    for aug in config_dict['input_features'][0]['augmentation']:
        assert isinstance(aug, dict)
    config_obj2 = ModelConfig.from_dict(config_dict)
    assert config_obj2.input_features[0].augmentation == config_obj.input_features[0].augmentation

@pytest.mark.parametrize('sequence_length, max_sequence_length, max_sequence_length_expected', [(None, 100, 100), (50, 100, 100), (100, 50, 100)])
def test_preprocessing_max_sequence_length(sequence_length, max_sequence_length, max_sequence_length_expected):
    if False:
        return 10
    config = {'input_features': [{'name': 'text1', 'type': 'text', 'preprocessing': {'sequence_length': sequence_length, 'max_sequence_length': max_sequence_length}}, {'name': 'sequence1', 'type': 'sequence', 'preprocessing': {'sequence_length': sequence_length, 'max_sequence_length': max_sequence_length}}], 'output_features': [{'name': 'number1', 'type': 'number'}]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].preprocessing.max_sequence_length == max_sequence_length_expected
    assert config_obj.input_features[1].preprocessing.max_sequence_length == max_sequence_length_expected

def test_gbm_encoders():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'feature_1', 'type': 'category'}, {'name': 'Sex', 'type': 'category'}], 'output_features': [{'name': 'Survived', 'type': 'category'}], 'defaults': {'binary': {'encoder': {'type': 'passthrough'}, 'preprocessing': {'missing_value_strategy': 'fill_with_false'}}, 'category': {'encoder': {'type': 'onehot'}, 'preprocessing': {'missing_value_strategy': 'fill_with_const', 'most_common': 10000}}, 'number': {'encoder': {'type': 'passthrough'}, 'preprocessing': {'missing_value_strategy': 'fill_with_const'}}}, 'model_type': 'gbm'}
    config_obj = ModelConfig.from_dict(config).to_dict()
    for feature_type in config_obj.get('defaults'):
        assert 'encoder' in config_obj['defaults'][feature_type]

def test_encoder_decoder_values_as_str():
    if False:
        i = 10
        return i + 15
    'Tests that encoder / decoder params provided as strings are properly converted to the correct type.'
    config = {'input_features': [{'name': 'text_input', 'type': 'text', 'encoder': 'bert'}], 'output_features': [{'name': 'cat_output', 'type': 'category', 'decoder': 'classifier'}]}
    config_obj = ModelConfig.from_dict(config)
    assert isinstance(config_obj.input_features[0].encoder, BERTConfig)
    assert isinstance(config_obj.output_features[0].decoder, ClassifierConfig)

@pytest.mark.parametrize('base_model_config,model_name', [('bloomz-3b', 'bigscience/bloomz-3b'), ('vicuna-7b', 'lmsys/vicuna-7b-v1.3'), ('huggyllama/llama-7b', 'huggyllama/llama-7b')])
def test_llm_base_model_config(base_model_config, model_name):
    if False:
        return 10
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: base_model_config, INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.base_model == model_name

@pytest.mark.parametrize('base_model_config', [None, 'invalid/model/name'])
def test_llm_base_model_config_error(base_model_config):
    if False:
        print('Hello World!')
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: base_model_config, INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}]}
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)

@pytest.mark.parametrize('bits,expected_qconfig', [(None, None), (4, QuantizationConfig(bits=4)), (8, QuantizationConfig(bits=8))])
def test_llm_quantization_config(bits: Optional[int], expected_qconfig: Optional[QuantizationConfig]):
    if False:
        return 10
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'bigscience/bloomz-3b', 'quantization': {'bits': bits}, INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}]}
    if bits is None:
        del config['quantization']
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.quantization == expected_qconfig

@pytest.mark.parametrize('rope_scaling_config', [{'type': 'linear'}, {'factor': 2.0}, {'type': 'linear', 'factor': 1.0}])
def test_llm_rope_scaling_failure_modes(rope_scaling_config: Union[None, Dict[str, Any]]):
    if False:
        while True:
            i = 10
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}], 'model_parameters': {'rope_scaling': rope_scaling_config}}
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)

def test_llm_model_parameters_no_rope_scaling():
    if False:
        print('Hello World!')
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}], 'model_parameters': {}}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.model_parameters.rope_scaling is None
    assert config_obj.model_parameters.to_dict() == {}

def test_llm_finetuning_output_feature_config():
    if False:
        for i in range(10):
            print('nop')
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'category_output', TYPE: 'category'}], 'trainer': {'type': 'finetune'}}
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)
    config[OUTPUT_FEATURES] = [{NAME: 'text_output', TYPE: 'text'}]
    ModelConfig.from_dict(config)

@pytest.mark.skip(reason='TODO(geoffrey, arnav): re-enable this when we have reconciled the config with the backend kwarg in api.py')
@pytest.mark.distributed
def test_llm_quantization_backend_compatibility():
    if False:
        return 10
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}], 'quantization': {'bits': 4}}
    ModelConfig.from_dict(config)
    config['backend'] = {'type': 'local'}
    ModelConfig.from_dict(config)
    config['backend'] = {'type': 'ray'}
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)
    import ray
    ray.init()
    config.pop('backend')
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)
    ray.shutdown()

class TestMaxNewTokensOverride:

    def test_max_new_tokens_override_no_changes_to_max_new_tokens(self):
        if False:
            i = 10
            return i + 15
        'Tests that the default value for max_new_tokens is respected when explicitly set in the config.'
        config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}], 'generation': {'max_new_tokens': 64}}
        config_obj = ModelConfig.from_dict(config)
        assert config_obj.generation.max_new_tokens == 64

    def test_max_new_tokens_override_large_max_sequence_length(self):
        if False:
            print('Hello World!')
        'Tests that the default value for max_new_tokens is overridden when max_sequence_length is set to a large\n        value than the default max_new_tokens.'
        config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text', 'preprocessing': {'max_sequence_length': 100}}]}
        config_obj = ModelConfig.from_dict(config)
        assert config_obj.generation.max_new_tokens == 100

    def test_max_new_tokens_override_large_global_max_sequence_length(self):
        if False:
            return 10
        'Tests that the default value for max_new_tokens is overridden when global_max_sequence_length is set to\n        a larger value than the default max_new_tokens.'
        config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}], PREPROCESSING: {'global_max_sequence_length': 100}}
        config_obj = ModelConfig.from_dict(config)
        assert config_obj.generation.max_new_tokens == 100

    def test_max_new_tokens_override_fallback_to_model_window_size(self):
        if False:
            print('Hello World!')
        config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [{NAME: 'text_input', TYPE: 'text'}], OUTPUT_FEATURES: [{NAME: 'text_output', TYPE: 'text'}]}
        config_obj = ModelConfig.from_dict(config)
        assert config_obj.generation.max_new_tokens == 1024