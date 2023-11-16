import pytest
from ludwig.config_validation.validation import check_schema, get_schema
from ludwig.constants import ACTIVE, AUDIO, BACKEND, CATEGORY, COLUMN, DECODER, DEFAULTS, ENCODER, LOSS, MODEL_ECD, MODEL_GBM, MODEL_LLM, MODEL_TYPE, NAME, PREPROCESSING, PROC_COLUMN, TRAINER, TYPE
from ludwig.error import ConfigValidationError
from ludwig.features.feature_registries import get_output_type_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.utils import get_combiner_jsonschema
from ludwig.schema.defaults.ecd import ECDDefaultsConfig
from ludwig.schema.defaults.gbm import GBMDefaultsConfig
from ludwig.schema.features.preprocessing.audio import AudioPreprocessingConfig
from ludwig.schema.features.preprocessing.bag import BagPreprocessingConfig
from ludwig.schema.features.preprocessing.binary import BinaryPreprocessingConfig
from ludwig.schema.features.preprocessing.category import CategoryPreprocessingConfig
from ludwig.schema.features.preprocessing.date import DatePreprocessingConfig
from ludwig.schema.features.preprocessing.h3 import H3PreprocessingConfig
from ludwig.schema.features.preprocessing.image import ImagePreprocessingConfig
from ludwig.schema.features.preprocessing.number import NumberPreprocessingConfig
from ludwig.schema.features.preprocessing.sequence import SequencePreprocessingConfig
from ludwig.schema.features.preprocessing.set import SetPreprocessingConfig
from ludwig.schema.features.preprocessing.text import TextPreprocessingConfig
from ludwig.schema.features.preprocessing.timeseries import TimeseriesPreprocessingConfig
from ludwig.schema.features.preprocessing.vector import VectorPreprocessingConfig
from ludwig.schema.features.utils import get_input_feature_jsonschema, get_output_feature_jsonschema
from ludwig.schema.llms.peft import LoraConfig
from ludwig.schema.model_types.base import ModelConfig
from ludwig.schema.utils import ludwig_dataclass, unload_jsonschema_from_marshmallow_class
from tests.integration_tests.utils import audio_feature, bag_feature, binary_feature, category_feature, date_feature, ENCODERS, h3_feature, image_feature, number_feature, sequence_feature, set_feature, text_feature, timeseries_feature, vector_feature

def test_config_features():
    if False:
        i = 10
        return i + 15
    all_input_features = [audio_feature('/tmp/destination_folder', encoder={'type': 'parallel_cnn'}), bag_feature(encoder={'type': 'embed'}), binary_feature(encoder={'type': 'passthrough'}), category_feature(encoder={'type': 'dense'}), date_feature(encoder={'type': 'embed'}), h3_feature(encoder={'type': 'embed'}), image_feature('/tmp/destination_folder', encoder={'type': 'stacked_cnn'}), number_feature(encoder={'type': 'passthrough'}), sequence_feature(encoder={'type': 'parallel_cnn'}), set_feature(encoder={'type': 'embed'}), text_feature(encoder={'type': 'parallel_cnn'}), timeseries_feature(encoder={'type': 'parallel_cnn'}), vector_feature(encoder={'type': 'dense'})]
    all_output_features = [binary_feature(decoder={'type': 'regressor'}), category_feature(decoder={'type': 'classifier'}), number_feature(decoder={'type': 'regressor'}), sequence_feature(decoder={'type': 'generator'}), set_feature(decoder={'type': 'classifier'}), text_feature(decoder={'type': 'generator'}), vector_feature(decoder={'type': 'projector'})]
    config = {'input_features': all_input_features, 'output_features': all_output_features}
    check_schema(config)
    input_only_features = [feature for feature in all_input_features if feature['type'] not in get_output_type_registry().keys()]
    for input_feature in input_only_features:
        config = {'input_features': all_input_features, 'output_features': all_output_features + [input_feature]}
        with pytest.raises(ConfigValidationError):
            check_schema(config)

def test_config_encoders():
    if False:
        for i in range(10):
            print('nop')
    for encoder in ENCODERS:
        config = {'input_features': [sequence_feature(encoder={'type': encoder, 'reduce_output': 'sum'}), image_feature('/tmp/destination_folder')], 'output_features': [category_feature(decoder={'type': 'classifier', 'vocab_size': 2}, reduce_input='sum')], 'combiner': {'type': 'concat', 'output_size': 14}}
        check_schema(config)

def test_config_with_backend():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet', 'size': 24, 'output_size': 26, 'sparsity': 1e-06, 'bn_virtual_divider': 32, 'bn_momentum': 0.4, 'num_steps': 5, 'relaxation_factor': 1.5, 'bn_virtual_bs': 512}, TRAINER: {'batch_size': 16384, 'eval_batch_size': 500000, 'epochs': 1000, 'early_stop': 20, 'learning_rate': 0.02, 'optimizer': {'type': 'adam'}, 'learning_rate_scheduler': {'decay': 'linear', 'decay_steps': 20000, 'decay_rate': 0.9, 'staircase': True}, 'regularization_lambda': 1, 'regularization_type': 'l2'}, BACKEND: {'type': 'ray', 'trainer': {'num_workers': 2}}}
    check_schema(config)

def test_config_bad_feature_type():
    if False:
        print('Hello World!')
    config = {'input_features': [{'name': 'foo', 'type': 'fake'}], 'output_features': [category_feature(encoder={'vocab_size': 2}, reduce_input='sum')], 'combiner': {'type': 'concat', 'output_size': 14}}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_config_bad_encoder_name():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [sequence_feature(encoder={'type': 'fake', 'reduce_output': 'sum'})], 'output_features': [category_feature(decoder={'type': 'classifier', 'vocab_size': 2}, reduce_input='sum')], 'combiner': {'type': 'concat', 'output_size': 14}}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_config_fill_values():
    if False:
        i = 10
        return i + 15
    vector_fill_values = ['1.0 0.0 1.04 10.49', '1 2 3 4 501.0']
    binary_fill_values = ['yes', 'No', '1', 'TRUE', 1]
    for (vector_fill_value, binary_fill_value) in zip(vector_fill_values, binary_fill_values):
        config = {'input_features': [vector_feature(preprocessing={'fill_value': vector_fill_value})], 'output_features': [binary_feature(preprocessing={'fill_value': binary_fill_value})]}
        check_schema(config)
    bad_vector_fill_values = ['one two three', '1,2,3', 0]
    bad_binary_fill_values = ['one', 2, 'maybe']
    for (vector_fill_value, binary_fill_value) in zip(bad_vector_fill_values, bad_binary_fill_values):
        config = {'input_features': [vector_feature(preprocessing={'fill_value': vector_fill_value})], 'output_features': [binary_feature(preprocessing={'fill_value': binary_fill_value})]}
        with pytest.raises(ConfigValidationError):
            check_schema(config)

def test_validate_with_preprocessing_defaults():
    if False:
        while True:
            i = 10
    config = {'input_features': [audio_feature('/tmp/destination_folder', preprocessing=AudioPreprocessingConfig().to_dict(), encoder={'type': 'parallel_cnn'}), bag_feature(preprocessing=BagPreprocessingConfig().to_dict(), encoder={'type': 'embed'}), binary_feature(preprocessing=BinaryPreprocessingConfig().to_dict(), encoder={'type': 'passthrough'}), category_feature(preprocessing=CategoryPreprocessingConfig().to_dict(), encoder={'type': 'dense'}), date_feature(preprocessing=DatePreprocessingConfig().to_dict(), encoder={'type': 'embed'}), h3_feature(preprocessing=H3PreprocessingConfig().to_dict(), encoder={'type': 'embed'}), image_feature('/tmp/destination_folder', preprocessing=ImagePreprocessingConfig().to_dict(), encoder={'type': 'stacked_cnn'}), number_feature(preprocessing=NumberPreprocessingConfig().to_dict(), encoder={'type': 'passthrough'}), sequence_feature(preprocessing=SequencePreprocessingConfig().to_dict(), encoder={'type': 'parallel_cnn'}), set_feature(preprocessing=SetPreprocessingConfig().to_dict(), encoder={'type': 'embed'}), text_feature(preprocessing=TextPreprocessingConfig().to_dict(), encoder={'type': 'parallel_cnn'}), timeseries_feature(preprocessing=TimeseriesPreprocessingConfig().to_dict(), encoder={'type': 'parallel_cnn'}), vector_feature(preprocessing=VectorPreprocessingConfig().to_dict(), encoder={'type': 'dense'})], 'output_features': [{'name': 'target', 'type': 'category'}], TRAINER: {'learning_rate_scheduler': {'decay': 'linear'}, 'learning_rate': 0.001, 'validation_field': 'target', 'validation_metric': 'accuracy'}}
    check_schema(config)

def test_ecd_defaults_schema():
    if False:
        return 10
    schema = ECDDefaultsConfig()
    assert schema.binary.decoder.type == 'regressor'
    assert schema.binary.encoder.type == 'passthrough'
    assert schema.category.encoder.dropout == 0.0
    assert ENCODER in schema.category.to_dict()
    assert PREPROCESSING in schema.category.to_dict()
    assert DECODER in schema.category.to_dict()
    assert LOSS in schema.category.to_dict()

def test_gbm_defaults_schema():
    if False:
        for i in range(10):
            print('nop')
    schema = GBMDefaultsConfig()
    assert AUDIO not in schema.to_dict()
    assert schema.binary.preprocessing.missing_value_strategy == 'fill_with_false'
    assert PREPROCESSING in schema.binary.to_dict()

def test_validate_defaults_schema():
    if False:
        i = 10
        return i + 15
    config = {'input_features': [category_feature(), number_feature()], 'output_features': [category_feature(output_feature=True)], 'defaults': {'category': {'preprocessing': {'missing_value_strategy': 'drop_row'}, 'encoder': {'type': 'sparse'}, 'decoder': {'type': 'classifier', 'norm_params': None, 'dropout': 0.0, 'use_bias': True}, 'loss': {'type': 'softmax_cross_entropy', 'confidence_penalty': 0}}, 'number': {'preprocessing': {'missing_value_strategy': 'fill_with_const', 'fill_value': 0}, 'loss': {'type': 'mean_absolute_error'}}}}
    check_schema(config)
    config[DEFAULTS][CATEGORY][NAME] = 'TEST'
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_validate_no_trainer_type():
    if False:
        print('Hello World!')
    config = {'model_type': 'ecd', 'input_features': [category_feature(), number_feature()], 'output_features': [category_feature(output_feature=True)], 'trainer': {'learning_rate': 'auto', 'batch_size': 'auto'}}
    check_schema(config)
    config[MODEL_TYPE] = MODEL_GBM
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER] = {'tree_learner': 'serial'}
    check_schema(config)
    config[MODEL_TYPE] = MODEL_ECD
    config[TRAINER] = {'tree_learner': 'serial'}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

def test_schema_no_duplicates():
    if False:
        print('Hello World!')
    schema = get_schema()
    popped_fields = [NAME, TYPE, COLUMN, PROC_COLUMN, ACTIVE]
    for field in popped_fields:
        assert field not in schema['properties']['input_features']['items']['allOf'][0]['then']['properties']
        assert field not in schema['properties']['output_features']['items']['allOf'][0]['then']['properties']
        assert field not in schema['properties']['combiner']['allOf'][0]['then']['properties']
        assert field not in schema['properties']['trainer']['properties']['optimizer']['allOf'][0]['then']['properties']
        assert field not in schema['properties']['input_features']['items']['allOf'][0]['then']['properties']['encoder']['allOf'][0]['then']['properties']
        assert field not in schema['properties']['output_features']['items']['allOf'][0]['then']['properties']['decoder']['allOf'][0]['then']['properties']

@pytest.mark.parametrize('model_type', [MODEL_ECD, MODEL_GBM, MODEL_LLM])
def test_ludwig_schema_serialization(model_type):
    if False:
        return 10
    import json
    schema = get_schema(model_type)
    try:
        json.dumps(schema)
    except TypeError as e:
        raise TypeError(f'Ludwig schema of type `{model_type}` cannot be represented by valid JSON. See further details: {e}')

def test_encoder_descriptions():
    if False:
        print('Hello World!')
    'This test tests that each encoder in the enum for each feature type has a description.'
    schema = get_input_feature_jsonschema(MODEL_ECD)
    for feature_schema in schema['allOf']:
        type_data = feature_schema['then']['properties']['encoder']['properties']['type']
        assert len(set(type_data['enumDescriptions'].keys())) > 0
        assert set(type_data['enumDescriptions'].keys()).issubset(set(type_data['enum']))

def test_combiner_descriptions():
    if False:
        while True:
            i = 10
    'This test tests that each combiner in the enum for available combiners has a description.'
    combiner_json_schema = get_combiner_jsonschema()
    type_data = combiner_json_schema['properties']['type']
    assert len(set(type_data['enumDescriptions'].keys())) > 0
    assert set(type_data['enumDescriptions'].keys()).issubset(set(type_data['enum']))

def test_decoder_descriptions():
    if False:
        print('Hello World!')
    'This test tests that each decoder in the enum for each feature type has a description.'
    schema = get_output_feature_jsonschema(MODEL_ECD)
    for feature_schema in schema['allOf']:
        type_data = feature_schema['then']['properties']['decoder']['properties']['type']
        assert len(type_data['enumDescriptions'].keys()) > 0
        assert set(type_data['enumDescriptions'].keys()).issubset(set(type_data['enum']))

def test_deprecation_warning_raised_for_unknown_parameters():
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [category_feature(encoder={'type': 'dense', 'vocab_size': 2}, reduce_input='sum'), number_feature()], 'output_features': [binary_feature()], 'combiner': {'type': 'tabnet', 'unknown_parameter_combiner': False}, TRAINER: {'epochs': 1000}}
    with pytest.warns(DeprecationWarning, match='not a valid parameter'):
        ModelConfig.from_dict(config)

@pytest.mark.parametrize('encoder_config,expected_adapter', [({'type': 'bert', 'trainable': True}, None), ({'type': 'bert', 'trainable': True, 'adapter': None}, None), ({'type': 'bert', 'trainable': True, 'adapter': {'type': 'lora'}}, LoraConfig()), ({'type': 'bert', 'trainable': True, 'adapter': {'type': 'lora', 'r': 16, 'alpha': 32, 'dropout': 0.1, 'bias_type': 'all'}}, LoraConfig(r=16, alpha=32, dropout=0.1, bias_type='all'))])
def test_text_encoder_adapter(encoder_config, expected_adapter):
    if False:
        for i in range(10):
            print('nop')
    config = {'input_features': [text_feature(encoder=encoder_config)], 'output_features': [category_feature(decoder={'type': 'classifier', 'vocab_size': 2}, reduce_input='sum')]}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].encoder.adapter == expected_adapter

def test_default_param_metadata():
    if False:
        print('Hello World!')

    @ludwig_dataclass
    class TestClass:
        test_schema_entry: str = schema_utils.StringOptions(options=['test'], default='test', description='')
    test_class = unload_jsonschema_from_marshmallow_class(TestClass)
    assert test_class['properties']['test_schema_entry']['parameter_metadata'] is not None