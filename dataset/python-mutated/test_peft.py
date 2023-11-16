import os
import pytest
from ludwig.constants import COMBINER, EPOCHS, INPUT_FEATURES, OUTPUT_FEATURES, TRAINER, TYPE
from tests.integration_tests.utils import binary_feature, generate_data, run_test_suite, text_feature

@pytest.mark.integration_tests_e
@pytest.mark.parametrize('backend', [pytest.param('local', id='local'), pytest.param('ray', id='ray', marks=pytest.mark.distributed)])
def test_text_adapter_lora(tmpdir, backend, ray_cluster_2cpu):
    if False:
        i = 10
        return i + 15
    input_features = [text_feature(encoder={'type': 'auto_transformer', 'pretrained_model_name_or_path': 'hf-internal-testing/tiny-bert-for-token-classification', 'trainable': True, 'adapter': {'type': 'lora'}})]
    output_features = [binary_feature()]
    data_csv_path = os.path.join(tmpdir, 'dataset.csv')
    dataset = generate_data(input_features, output_features, data_csv_path)
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, COMBINER: {TYPE: 'concat', 'output_size': 14}, TRAINER: {EPOCHS: 1}}
    model = run_test_suite(config, dataset, backend)
    state_dict = model.model.state_dict()
    assert any(('lora_' in key for key in state_dict.keys()))