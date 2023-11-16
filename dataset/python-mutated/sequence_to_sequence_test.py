import os, sys
import numpy as np
from cntk import load_model
from cntk.device import try_set_default_device
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'SequenceToSequence', 'CMUDict', 'Python'))
TOLERANCE_ABSOLUTE = 0.01

def test_sequence_to_sequence(device_id):
    if False:
        print('Hello World!')
    from Sequence2Sequence import create_reader, DATA_DIR, MODEL_DIR, TRAINING_DATA, VALIDATION_DATA, TESTING_DATA, VOCAB_FILE, get_vocab, create_model, model_path_stem, train, evaluate_metric
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    train_reader = create_reader(os.path.join(DATA_DIR, TRAINING_DATA), False)
    valid_reader = create_reader(os.path.join(DATA_DIR, VALIDATION_DATA), True)
    test_reader = create_reader(os.path.join(DATA_DIR, TESTING_DATA), False)
    (vocab, i2w, _) = get_vocab(os.path.join(DATA_DIR, VOCAB_FILE))
    model = create_model()
    train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1, epoch_size=5000)
    model_filename = os.path.join(MODEL_DIR, model_path_stem + '.cmf.0')
    model = load_model(model_filename)
    error = evaluate_metric(test_reader, model, 10)
    print(error)
    expected_error = 0.95
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)