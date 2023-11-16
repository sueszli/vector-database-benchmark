import pytest

from ludwig.modules.reduction_modules import reduce_mode_registry
from tests.integration_tests.utils import category_feature, generate_data, run_experiment, sequence_feature


@pytest.mark.parametrize("reduce_output", reduce_mode_registry)
def test_reduction(reduce_output, csv_filename):
    input_features = [sequence_feature(reduce_output=reduce_output)]

    output_features = [category_feature(output_feature=True)]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)

    del input_features
    del output_features
