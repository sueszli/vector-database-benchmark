import contextlib
import copy
from io import StringIO
import pandas as pd
import pytest
import torch
from ludwig.api import LudwigModel
from ludwig.constants import DECODER, ENCODER_OUTPUT_STATE, LOGITS
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from tests.integration_tests.utils import generate_data, run_experiment, sequence_feature
TEST_VOCAB_SIZE = 132
TEST_HIDDEN_SIZE = 32
TEST_STATE_SIZE = 8
TEST_EMBEDDING_SIZE = 64
TEST_NUM_FILTERS = 24

@pytest.fixture(scope='module')
def generate_sequence_training_data():
    if False:
        for i in range(10):
            print('nop')
    input_features = [sequence_feature(encoder={'vocab_size': TEST_VOCAB_SIZE, 'embedding_size': TEST_EMBEDDING_SIZE, 'state_size': TEST_STATE_SIZE, 'hidden_size': TEST_HIDDEN_SIZE, 'num_filters': TEST_NUM_FILTERS, 'min_len': 5, 'max_len': 10, 'type': 'rnn', 'cell_type': 'lstm'})]
    output_features = [sequence_feature(decoder={'type': 'generator', 'min_len': 5, 'max_len': 10, 'cell_type': 'lstm', 'attention': 'bahdanau'})]
    dataset = build_synthetic_dataset(150, copy.deepcopy(input_features) + copy.deepcopy(output_features))
    raw_data = '\n'.join([r[0] + ',' + r[1] for r in dataset])
    df = pd.read_csv(StringIO(raw_data))
    return (df, input_features, output_features)

@contextlib.contextmanager
def setup_model_scaffolding(raw_df, input_features, output_features):
    if False:
        return 10
    config = {'input_features': input_features, 'output_features': output_features}
    model = LudwigModel(config)
    (training_set, _, _, training_set_metadata) = preprocess_for_training(model.config, training_set=raw_df, skip_save_processed_input=True)
    model.training_set_metadata = training_set_metadata
    update_config_with_metadata(model.config_obj, training_set_metadata)
    model.model = model.create_model(model.config_obj)
    with training_set.initialize_batcher() as batcher:
        yield (model, batcher)

@pytest.mark.parametrize('dec_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('combiner_output_shapes', [((128, 10, TEST_STATE_SIZE), None), ((128, 10, TEST_STATE_SIZE), ((128, TEST_STATE_SIZE), (128, TEST_STATE_SIZE))), ((128, 10, TEST_STATE_SIZE), ((128, TEST_STATE_SIZE),))])
def test_sequence_decoders(dec_cell_type, combiner_output_shapes, generate_sequence_training_data):
    if False:
        for i in range(10):
            print('nop')
    raw_df = generate_sequence_training_data[0]
    input_features = generate_sequence_training_data[1]
    output_features = generate_sequence_training_data[2]
    output_feature_name = output_features[0]['name']
    output_features[0][DECODER]['cell_type'] = dec_cell_type
    with setup_model_scaffolding(raw_df, input_features, output_features) as (model, _):
        encoder_output = torch.randn(combiner_output_shapes[0])
        combiner_outputs = {'hidden': encoder_output}
        if combiner_output_shapes[1] is not None:
            if len(combiner_output_shapes[1]) > 1:
                encoder_output_state = (torch.randn(combiner_output_shapes[1][0]), torch.randn(combiner_output_shapes[1][1]))
            else:
                encoder_output_state = torch.randn(combiner_output_shapes[1][0])
            combiner_outputs[ENCODER_OUTPUT_STATE] = encoder_output_state
        decoder = model.model.output_features.get(output_feature_name).decoder_obj
        decoder_out = decoder(combiner_outputs)
        batch_size = combiner_outputs['hidden'].shape[0]
        seq_size = output_features[0][DECODER]['max_len'] + 2
        vocab_size = model.config_obj.output_features.to_list()[0][DECODER]['vocab_size']
        assert list(decoder_out[LOGITS].size()) == [batch_size, seq_size, vocab_size]

@pytest.mark.parametrize('dec_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('enc_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('enc_encoder', ['embed', 'rnn'])
def test_sequence_generator(enc_encoder, enc_cell_type, dec_cell_type, csv_filename):
    if False:
        print('Hello World!')
    input_features = [sequence_feature(encoder={'type': enc_encoder, 'min_len': 5, 'max_len': 10, 'cell_type': enc_cell_type})]
    output_features = [sequence_feature(decoder={'type': 'generator', 'min_len': 5, 'max_len': 10, 'cell_type': dec_cell_type})]
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)