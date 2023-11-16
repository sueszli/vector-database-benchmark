import logging
import os
import pandas as pd
import pytest
from ludwig.constants import NAME
from tests.integration_tests.utils import bag_feature, binary_feature, category_feature, generate_data, number_feature, run_experiment, sequence_feature, set_feature, text_feature, vector_feature
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('ludwig').setLevel(logging.INFO)

@pytest.mark.parametrize('input_test_feature, output_test_feature, output_loss_parameter', [(number_feature(), number_feature(), None), (number_feature(normalization='minmax'), number_feature(), {'loss': {'type': 'mean_squared_error'}}), (number_feature(normalization='zscore'), number_feature(), {'loss': {'type': 'mean_absolute_error'}}), (binary_feature(), binary_feature(), None), (category_feature(), category_feature(output_feature=True), None), (category_feature(), category_feature(output_feature=True), {'loss': {'type': 'softmax_cross_entropy'}})])
def test_feature(input_test_feature, output_test_feature, output_loss_parameter, csv_filename):
    if False:
        i = 10
        return i + 15
    input_features = [input_test_feature]
    of_test_feature = output_test_feature
    if output_loss_parameter is not None:
        of_test_feature.update(output_loss_parameter)
    output_features = [of_test_feature]
    rel_path = generate_data(input_features, output_features, csv_filename, 1001)
    run_experiment(input_features, output_features, dataset=rel_path)

@pytest.mark.parametrize('input_test_feature, output_test_feature', [([category_feature()], [binary_feature(), binary_feature()]), ([category_feature()], [category_feature(decoder={'vocab_size': 5}), category_feature(decoder={'vocab_size': 7})]), ([category_feature()], [number_feature(), number_feature()]), ([category_feature()], [sequence_feature(decoder={'vocab_size': 5}), sequence_feature(decoder={'vocab_size': 7})]), ([set_feature(encoder={'vocab_size': 5})], [set_feature(decoder={'vocab_size': 5}), set_feature(decoder={'vocab_size': 7})]), ([category_feature()], [text_feature(decoder={'vocab_size': 5}), text_feature(decoder={'vocab_size': 7})]), ([category_feature()], [vector_feature(), vector_feature()]), ([vector_feature()], [vector_feature(), vector_feature()]), ([bag_feature()], [vector_feature(), vector_feature()])])
def test_feature_multiple_outputs(input_test_feature, output_test_feature, csv_filename):
    if False:
        while True:
            i = 10
    rel_path = generate_data(input_test_feature, output_test_feature, csv_filename, 1001)
    run_experiment(input_test_feature, output_test_feature, dataset=rel_path)

def test_category_int_dtype(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    feature = category_feature()
    input_features = [feature]
    output_features = [binary_feature()]
    csv_fname = generate_data(input_features, output_features, os.path.join(tmpdir, 'dataset.csv'))
    df = pd.read_csv(csv_fname)
    distinct_values = df[feature[NAME]].drop_duplicates().values
    value_map = {v: idx for (idx, v) in enumerate(distinct_values)}
    df[feature[NAME]] = df[feature[NAME]].map(lambda x: value_map[x])
    run_experiment(input_features, output_features, dataset=df)