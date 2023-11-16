import pytest
from ludwig.config_validation.preprocessing import check_global_max_sequence_length_fits_prompt_template
from ludwig.config_validation.validation import check_schema
from tests.integration_tests.utils import binary_feature, category_feature

def test_config_preprocessing():
    if False:
        return 10
    input_features = [category_feature(), category_feature()]
    output_features = [binary_feature()]
    config = {'input_features': input_features, 'output_features': output_features, 'preprocessing': {'split': {'type': 'random', 'probabilities': [0.6, 0.2, 0.2]}, 'oversample_minority': 0.4}}
    check_schema(config)

def test_check_global_max_sequence_length_fits_prompt_template():
    if False:
        print('Hello World!')
    check_global_max_sequence_length_fits_prompt_template({'input_feature': {'prompt_template_num_tokens': 10}}, {'global_max_sequence_length': 10})
    check_global_max_sequence_length_fits_prompt_template({'input_feature': {'prompt_template_num_tokens': 100}}, {'global_max_sequence_length': 1000})
    check_global_max_sequence_length_fits_prompt_template({'input_feature': {'prompt_template_num_tokens': 100}}, {'global_max_sequence_length': None})
    with pytest.raises(ValueError):
        check_global_max_sequence_length_fits_prompt_template({'input_feature': {'prompt_template_num_tokens': 10}}, {'global_max_sequence_length': 5})
    with pytest.raises(ValueError):
        check_global_max_sequence_length_fits_prompt_template({'input_feature': {'prompt_template_num_tokens': 5}, 'input_feature_2': {'prompt_template_num_tokens': 20}}, {'global_max_sequence_length': 10})