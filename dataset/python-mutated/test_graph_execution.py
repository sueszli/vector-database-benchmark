import pytest
from tests.integration_tests.utils import category_feature, generate_data, generate_output_features_with_dependencies, number_feature, run_experiment, sequence_feature, set_feature, text_feature

@pytest.mark.parametrize('output_features', [[category_feature(decoder={'vocab_size': 2}, reduce_input='sum'), sequence_feature(decoder={'vocab_size': 10, 'max_len': 5}), number_feature()], [category_feature(decoder={'vocab_size': 2}, reduce_input='sum'), sequence_feature(decoder={'vocab_size': 10, 'max_len': 5, 'type': 'generator'}), number_feature()], [category_feature(decoder={'vocab_size': 2}, reduce_input='sum'), sequence_feature(decoder={'max_len': 5, 'type': 'generator'}, reduce_input=None), number_feature(normalization='minmax')], generate_output_features_with_dependencies('number_feature', ['category_feature']), generate_output_features_with_dependencies('number_feature', ['category_feature', 'sequence_feature'])])
def test_experiment_multiple_seq_seq(csv_filename, output_features):
    if False:
        while True:
            i = 10
    input_features = [text_feature(encoder={'vocab_size': 100, 'min_len': 1, 'type': 'stacked_cnn'}), number_feature(normalization='zscore'), category_feature(encoder={'vocab_size': 10, 'embedding_size': 5}), set_feature(), sequence_feature(encoder={'vocab_size': 10, 'max_len': 10, 'type': 'embed'})]
    output_features = output_features
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)