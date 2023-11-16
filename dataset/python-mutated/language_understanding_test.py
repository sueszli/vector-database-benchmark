from __future__ import print_function
import os, sys
import numpy as np
import shutil
from cntk import DeviceDescriptor
TOLERANCE_ABSOLUTE = 0.1
from cntk import placeholder
from cntk.layers import *
from cntk.internal.utils import *
from cntk.logging import *
from cntk.ops import splice
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'LanguageUnderstanding', 'ATIS', 'Python'))
sys.path.append('../LanguageUnderstanding/ATIS/Python')
from LanguageUnderstanding import data_dir, create_reader, create_model_function, train, evaluate, emb_dim, hidden_dim, num_labels

def run_model_test(what, model, expected_train):
    if False:
        while True:
            i = 10
    print('--- {} ---'.format(what))
    reader = create_reader(data_dir + '/atis.train.ctf', is_training=True)
    (loss, metric) = train(reader, model, max_epochs=1)
    print('-->', metric, loss)
    assert np.allclose([metric, loss], expected_train, atol=TOLERANCE_ABSOLUTE)

def create_test_model():
    if False:
        print('Hello World!')
    with default_options(enable_self_stabilization=True, use_peepholes=True):
        return Sequential([Embedding(emb_dim), BatchNormalization(), Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim + 50), go_backwards=True), BatchNormalization(map_rank=1), Dense(num_labels)])

def with_lookahead():
    if False:
        return 10
    x = placeholder()
    future_x = sequence.future_value(x)
    apply_x = splice(x, future_x)
    return apply_x

def BiRecurrence(fwd, bwd):
    if False:
        print('Hello World!')
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    x = placeholder()
    apply_x = splice(F(x), G(x))
    return apply_x

def BNBiRecurrence(fwd, bwd, test_dual=True):
    if False:
        while True:
            i = 10
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    BN = BatchNormalization(normalization_time_constant=-1)
    x = placeholder()
    x1 = BN(x)
    x2 = BN(x) if test_dual else x1
    apply_x = splice(F(x1), G(x2))
    return apply_x

def test_language_understanding(device_id):
    if False:
        return 10
    from cntk.ops.tests.ops_test_utils import cntk_device
    DeviceDescriptor.try_set_default_device(cntk_device(device_id))
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)
    force_deterministic_algorithms()
    if device_id >= 0:
        with default_options(initial_state=0.1):
            run_model_test('replace lookahead by bidirectional model', Sequential([Embedding(emb_dim), BatchNormalization(), BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)), BatchNormalization(), Dense(num_labels)]), [0.0579573500457558, 0.3214986774820327])
        with default_options(initial_state=0.1):
            run_model_test('replace lookahead by bidirectional model, with shared BN', Sequential([Embedding(emb_dim), BNBiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim), test_dual=True), BatchNormalization(normalization_time_constant=-1), Dense(num_labels)]), [0.0579573500457558, 0.3214986774820327])
            ' with normalization_time_constant=-1:\n             Minibatch[   1-   1]: loss = 5.945220 * 67, metric = 100.0% * 67\n             Minibatch[   2-   2]: loss = 4.850601 * 63, metric = 79.4% * 63\n             Minibatch[   3-   3]: loss = 3.816031 * 68, metric = 57.4% * 68\n             Minibatch[   4-   4]: loss = 2.213172 * 70, metric = 41.4% * 70\n             Minibatch[   5-   5]: loss = 2.615342 * 65, metric = 40.0% * 65\n             Minibatch[   6-   6]: loss = 2.360896 * 62, metric = 25.8% * 62\n             Minibatch[   7-   7]: loss = 1.452822 * 58, metric = 27.6% * 58\n             Minibatch[   8-   8]: loss = 0.947210 * 70, metric = 10.0% * 70\n             Minibatch[   9-   9]: loss = 0.595654 * 59, metric = 10.2% * 59\n             Minibatch[  10-  10]: loss = 1.515479 * 64, metric = 23.4% * 64\n             Minibatch[  11- 100]: loss = 0.686744 * 5654, metric = 10.4% * 5654\n             Minibatch[ 101- 200]: loss = 0.289059 * 6329, metric = 5.8% * 6329\n             Minibatch[ 201- 300]: loss = 0.218765 * 6259, metric = 4.7% * 6259\n             Minibatch[ 301- 400]: loss = 0.182855 * 6229, metric = 3.5% * 6229\n             Minibatch[ 401- 500]: loss = 0.156745 * 6289, metric = 3.4% * 6289\n            Finished Epoch [1]: [Training] loss = 0.321413 * 36061, metric = 5.8% * 36061\n            --> 0.057818696098277916 0.3214128415043278\n             Minibatch[   1-   1]: loss = 0.000000 * 991, metric = 2.5% * 991\n             Minibatch[   2-   2]: loss = 0.000000 * 1000, metric = 2.8% * 1000\n             Minibatch[   3-   3]: loss = 0.000000 * 992, metric = 4.0% * 992\n             Minibatch[   4-   4]: loss = 0.000000 * 989, metric = 3.0% * 989\n             Minibatch[   5-   5]: loss = 0.000000 * 998, metric = 3.8% * 998\n             Minibatch[   6-   6]: loss = 0.000000 * 995, metric = 1.5% * 995\n             Minibatch[   7-   7]: loss = 0.000000 * 998, metric = 2.5% * 998\n             Minibatch[   8-   8]: loss = 0.000000 * 992, metric = 1.6% * 992\n             Minibatch[   9-   9]: loss = 0.000000 * 1000, metric = 1.6% * 1000\n             Minibatch[  10-  10]: loss = 0.000000 * 996, metric = 7.9% * 996\n            Finished Epoch [1]: [Evaluation] loss = 0.000000 * 10984, metric = 3.2% * 10984\n            --> 0.03159140568099053 0.0\n            '
        with default_options(initial_state=0.1):
            run_model_test('BatchNorm global-corpus aggregation', Sequential([Embedding(emb_dim), BatchNormalization(normalization_time_constant=-1), Recurrence(LSTM(hidden_dim), go_backwards=False), BatchNormalization(normalization_time_constant=-1), Dense(num_labels)]), [0.05662627214996811, 0.2968516879905391])
            '\n             Minibatch[   1-   1]: loss = 5.745576 * 67, metric = 100.0% * 67\n             Minibatch[   2-   2]: loss = 4.684151 * 63, metric = 90.5% * 63\n             Minibatch[   3-   3]: loss = 3.957423 * 68, metric = 63.2% * 68\n             Minibatch[   4-   4]: loss = 2.286908 * 70, metric = 41.4% * 70\n             Minibatch[   5-   5]: loss = 2.733978 * 65, metric = 38.5% * 65\n             Minibatch[   6-   6]: loss = 2.189765 * 62, metric = 30.6% * 62\n             Minibatch[   7-   7]: loss = 1.427890 * 58, metric = 25.9% * 58\n             Minibatch[   8-   8]: loss = 1.501557 * 70, metric = 18.6% * 70\n             Minibatch[   9-   9]: loss = 0.632599 * 59, metric = 13.6% * 59\n             Minibatch[  10-  10]: loss = 1.516047 * 64, metric = 23.4% * 64\n             Minibatch[  11- 100]: loss = 0.580329 * 5654, metric = 9.8% * 5654\n             Minibatch[ 101- 200]: loss = 0.280317 * 6329, metric = 5.6% * 6329\n             Minibatch[ 201- 300]: loss = 0.188372 * 6259, metric = 4.1% * 6259\n             Minibatch[ 301- 400]: loss = 0.170403 * 6229, metric = 3.9% * 6229\n             Minibatch[ 401- 500]: loss = 0.159605 * 6289, metric = 3.4% * 6289\n            Finished Epoch [1]: [Training] loss = 0.296852 * 36061, metric = 5.7% * 36061\n            --> 0.05662627214996811 0.2968516879905391\n             Minibatch[   1-   1]: loss = 0.000000 * 991, metric = 1.8% * 991\n             Minibatch[   2-   2]: loss = 0.000000 * 1000, metric = 3.4% * 1000\n             Minibatch[   3-   3]: loss = 0.000000 * 992, metric = 3.9% * 992\n             Minibatch[   4-   4]: loss = 0.000000 * 989, metric = 4.1% * 989\n             Minibatch[   5-   5]: loss = 0.000000 * 998, metric = 4.0% * 998\n             Minibatch[   6-   6]: loss = 0.000000 * 995, metric = 1.2% * 995\n             Minibatch[   7-   7]: loss = 0.000000 * 998, metric = 2.8% * 998\n             Minibatch[   8-   8]: loss = 0.000000 * 992, metric = 2.9% * 992\n             Minibatch[   9-   9]: loss = 0.000000 * 1000, metric = 2.0% * 1000\n             Minibatch[  10-  10]: loss = 0.000000 * 996, metric = 8.2% * 996\n            Finished Epoch [1]: [Evaluation] loss = 0.000000 * 10984, metric = 3.5% * 10984\n            --> 0.035050983248361256 0.0\n            '
        with default_options(initial_state=0.1):
            run_model_test('plus BatchNorm', Sequential([Embedding(emb_dim), BatchNormalization(), Recurrence(LSTM(hidden_dim), go_backwards=False), BatchNormalization(), Dense(num_labels)]), [0.05662627214996811, 0.2968516879905391])
        with default_options(initial_state=0.1):
            run_model_test('plus lookahead', Sequential([Embedding(emb_dim), with_lookahead(), BatchNormalization(), Recurrence(LSTM(hidden_dim), go_backwards=False), BatchNormalization(), Dense(num_labels)]), [0.057901888466764646, 0.3044637752807047])
        with default_options(initial_state=0.1):
            run_model_test('replace lookahead by bidirectional model', Sequential([Embedding(emb_dim), BatchNormalization(), BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)), BatchNormalization(), Dense(num_labels)]), [0.0579573500457558, 0.3214986774820327])
        with default_options(enable_self_stabilization=True, use_peepholes=True):
            run_model_test('alternate paths', Sequential([Embedding(emb_dim), BatchNormalization(), Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim + 50), go_backwards=True), BatchNormalization(map_rank=1), Dense(num_labels)]), [0.08574360112032389, 0.41847621578367716])
    if device_id >= 0:
        reader = create_reader(data_dir + '/atis.train.ctf', is_training=True)
        model = create_model_function()
        (loss_avg, evaluation_avg) = train(reader, model, max_epochs=1)
        expected_avg = [0.09698114255561419, 0.5290531086061565]
        assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
        reader = create_reader(data_dir + '/atis.test.ctf', is_training=False)
        evaluate(reader, model)
    if device_id >= 0:
        abs_path = os.path.dirname(os.path.abspath(__file__))
        tb_logdir = os.path.join(abs_path, 'language_understanding_test_log')
        if os.path.exists(tb_logdir):
            shutil.rmtree(tb_logdir)
        reader = create_reader(data_dir + '/atis.train.ctf', is_training=True)
        model = create_test_model()
        (loss_avg, evaluation_avg) = train(reader, model, max_epochs=1)
        log_number_of_parameters(model, trace_level=1)
        print()
        expected_avg = [0.084, 0.407364]
        assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
if __name__ == '__main__':
    test_language_understanding(0)