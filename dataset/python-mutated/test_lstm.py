"""
This test compares the NEON LSTM layer against a numpy reference LSTM
implementation and compares the NEON LSTM bprop deltas to the gradients
estimated by finite differences.
The numpy reference LSTM contains static methods for forward pass
and backward pass.
It runs a SINGLE layer of LSTM and compare numerical values

The following are made sure to be the same in both LSTMs
    -   initial c, h values (all zeros)
    -   initial W, b (random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside LSTM_np is seq_len, batch_size, input_size.
        Need transpose
    -   the data shape inside LSTM (neon) is input_size, seq_len * batch_size

"""
import itertools as itt
import numpy as np
from neon import NervanaObject, logger as neon_logger
from neon.initializers.initializer import Constant, Gaussian
from neon.layers.recurrent import LSTM
from neon.layers.container import DeltasTree
from neon.transforms import Logistic, Tanh
from lstm_ref import LSTM as RefLSTM
from utils import sparse_rand, allclose_with_out

def pytest_generate_tests(metafunc):
    if False:
        return 10
    if metafunc.config.option.all:
        bsz_rng = [16, 32]
    else:
        bsz_rng = [16]
    if 'reflstmargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4]
            inp_rng = [3, 5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('reflstmargs', fargs)
    if 'gradlstmargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3]
            inp_rng = [5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('gradlstmargs', fargs)

def test_ref_compare_ones(backend_default, reflstmargs):
    if False:
        return 10
    np.random.seed(seed=0)
    (seq_len, input_size, hidden_size, batch_size) = reflstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    check_lstm(seq_len, input_size, hidden_size, batch_size, Constant(val=1.0), [1.0, 0.0])

def test_ref_compare_rand(backend_default, reflstmargs):
    if False:
        print('Hello World!')
    np.random.seed(seed=0)
    (seq_len, input_size, hidden_size, batch_size) = reflstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    check_lstm(seq_len, input_size, hidden_size, batch_size, Gaussian())

def check_lstm(seq_len, input_size, hidden_size, batch_size, init_func, inp_moms=[0.0, 1.0]):
    if False:
        while True:
            i = 10
    input_shape = (input_size, seq_len * batch_size)
    hidden_shape = (hidden_size, seq_len * batch_size)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    lstm = LSTM(hidden_size, init_func, activation=Tanh(), gate_activation=Logistic())
    inp = np.random.rand(*input_shape) * inp_moms[1] + inp_moms[0]
    inpa = lstm.be.array(inp)
    lstm.configure((input_size, seq_len))
    lstm.prev_layer = True
    lstm.allocate()
    dtree = DeltasTree()
    lstm.allocate_deltas(dtree)
    dtree.allocate_buffers()
    lstm.set_deltas(dtree)
    lstm.fprop(inpa)
    lstm_ref = RefLSTM()
    WLSTM = lstm_ref.init(input_size, hidden_size)
    WLSTM[0, :] = lstm.b.get().T
    WLSTM[1:input_size + 1, :] = lstm.W_input.get().T
    WLSTM[input_size + 1:] = lstm.W_recur.get().T
    inp_ref = inp.copy().T.reshape(seq_len, batch_size, input_size)
    (Hout_ref, cprev, hprev, batch_cache) = lstm_ref.forward(inp_ref, WLSTM)
    Hout_ref = Hout_ref.reshape(seq_len * batch_size, hidden_size).T
    IFOGf_ref = batch_cache['IFOGf'].reshape(seq_len * batch_size, hidden_size * 4).T
    Ct_ref = batch_cache['Ct'].reshape(seq_len * batch_size, hidden_size).T
    neon_logger.display('====Verifying IFOG====')
    assert allclose_with_out(lstm.ifog_buffer.get(), IFOGf_ref, rtol=0.0, atol=1.5e-05)
    neon_logger.display('====Verifying cell states====')
    assert allclose_with_out(lstm.c_act_buffer.get(), Ct_ref, rtol=0.0, atol=1.5e-05)
    neon_logger.display('====Verifying hidden states====')
    assert allclose_with_out(lstm.outputs.get(), Hout_ref, rtol=0.0, atol=1.5e-05)
    neon_logger.display('fprop is verified')
    deltas = np.random.randn(*hidden_shape)
    lstm.bprop(lstm.be.array(deltas))
    dWinput_neon = lstm.dW_input.get()
    dWrecur_neon = lstm.dW_recur.get()
    db_neon = lstm.db.get()
    deltas_ref = deltas.copy().T.reshape(seq_len, batch_size, hidden_size)
    (dX_ref, dWLSTM_ref, dc0_ref, dh0_ref) = lstm_ref.backward(deltas_ref, batch_cache)
    dWrecur_ref = dWLSTM_ref[-hidden_size:, :]
    dWinput_ref = dWLSTM_ref[1:input_size + 1, :]
    db_ref = dWLSTM_ref[0, :]
    dX_ref = dX_ref.reshape(seq_len * batch_size, input_size).T
    neon_logger.display('Making sure neon LSTM match numpy LSTM in bprop')
    neon_logger.display('====Verifying update on W_recur====')
    assert allclose_with_out(dWrecur_neon, dWrecur_ref.T, rtol=0.0, atol=1.5e-05)
    neon_logger.display('====Verifying update on W_input====')
    assert allclose_with_out(dWinput_neon, dWinput_ref.T, rtol=0.0, atol=1.5e-05)
    neon_logger.display('====Verifying update on bias====')
    assert allclose_with_out(db_neon.flatten(), db_ref, rtol=0.0, atol=1.5e-05)
    neon_logger.display('====Verifying output delta====')
    assert allclose_with_out(lstm.out_deltas_buffer.get(), dX_ref, rtol=0.0, atol=1.5e-05)
    neon_logger.display('bprop is verified')
    return

def reset_lstm(lstm):
    if False:
        print('Hello World!')
    lstm.x = None
    lstm.xs = None
    lstm.outputs = None
    return

def test_gradient_ref_lstm(backend_default, gradlstmargs):
    if False:
        print('Hello World!')
    (seq_len, input_size, hidden_size, batch_size) = gradlstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check_ref(seq_len, input_size, hidden_size, batch_size)

def test_gradient_neon_lstm(backend_default, gradlstmargs):
    if False:
        print('Hello World!')
    (seq_len, input_size, hidden_size, batch_size) = gradlstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size)

def gradient_check_ref(seq_len, input_size, hidden_size, batch_size, epsilon=1e-05, dtypeu=np.float64, threshold=0.0001):
    if False:
        return 10
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    input_shape = (seq_len, input_size, batch_size)
    (inp_bl, nz_inds) = sparse_rand(input_shape, frac=1.0 / float(input_shape[1]))
    inp_bl = np.random.randn(*input_shape)
    inp_bl = inp_bl.swapaxes(1, 2).astype(dtypeu)
    lstm_ref = RefLSTM()
    WLSTM = lstm_ref.init(input_size, hidden_size).astype(dtypeu)
    WLSTM = np.random.randn(*WLSTM.shape)
    (Hout, cprev, hprev, cache) = lstm_ref.forward(inp_bl, WLSTM)
    rand_scale = np.random.random(Hout.shape) * 2.0 - 1.0
    rand_scale = dtypeu(rand_scale)
    (dX_bl, dWLSTM_bl, dc0, dh0) = lstm_ref.backward(rand_scale, cache)
    grads_est = np.zeros(dX_bl.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inp_bl.size):
        save_val = inp_pert.flat[pert_ind]
        inp_pert.flat[pert_ind] = save_val + epsilon
        (Hout_pos, cprev, hprev, cache) = lstm_ref.forward(inp_pert, WLSTM)
        inp_pert.flat[pert_ind] = save_val - epsilon
        (Hout_neg, cprev, hprev, cache) = lstm_ref.forward(inp_pert, WLSTM)
        loss_pos = np.sum(rand_scale * Hout_pos)
        loss_neg = np.sum(rand_scale * Hout_neg)
        grads_est.flat[pert_ind] = 0.5 / float(epsilon) * (loss_pos - loss_neg)
        inp_pert.flat[pert_ind] = save_val
    assert allclose_with_out(grads_est, dX_bl, rtol=threshold, atol=0.0)
    return

def gradient_check(seq_len, input_size, hidden_size, batch_size, threshold=0.001):
    if False:
        return 10
    min_max_err = -1.0
    neon_logger.display('Perturb mag, max grad diff')
    for pert_exp in range(-5, 0):
        input_shape = (input_size, seq_len * batch_size)
        output_shape = (hidden_size, seq_len * batch_size)
        rand_scale = np.random.random(output_shape) * 2.0 - 1.0
        inp = np.random.randn(*input_shape)
        pert_mag = 10.0 ** pert_exp
        (grad_est, deltas) = gradient_calc(seq_len, input_size, hidden_size, batch_size, epsilon=pert_mag, rand_scale=rand_scale, inp_bl=inp)
        dd = np.max(np.abs(grad_est - deltas))
        neon_logger.display('%e, %e' % (pert_mag, dd))
        if min_max_err < 0.0 or dd < min_max_err:
            min_max_err = dd
        NervanaObject.be.rng_reset()
    neon_logger.display('Worst case error %e with perturbation %e' % (min_max_err, pert_mag))
    neon_logger.display('Threshold %e' % threshold)
    assert min_max_err < threshold

def gradient_calc(seq_len, input_size, hidden_size, batch_size, epsilon=None, rand_scale=None, inp_bl=None):
    if False:
        print('Hello World!')
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    input_shape = (input_size, seq_len * batch_size)
    if inp_bl is None:
        inp_bl = np.random.randn(*input_shape)
    lstm = LSTM(hidden_size, Gaussian(), activation=Tanh(), gate_activation=Logistic())
    inpa = lstm.be.array(np.copy(inp_bl))
    lstm.configure((input_size, seq_len))
    lstm.prev_layer = True
    lstm.allocate()
    dtree = DeltasTree()
    lstm.allocate_deltas(dtree)
    dtree.allocate_buffers()
    lstm.set_deltas(dtree)
    out_bl = lstm.fprop(inpa).get()
    if rand_scale is None:
        rand_scale = np.random.random(out_bl.shape) * 2.0 - 1.0
    deltas_neon = lstm.bprop(lstm.be.array(np.copy(rand_scale))).get()
    grads_est = np.zeros(inpa.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inpa.size):
        save_val = inp_pert.flat[pert_ind]
        inp_pert.flat[pert_ind] = save_val + epsilon
        reset_lstm(lstm)
        lstm.allocate()
        out_pos = lstm.fprop(lstm.be.array(inp_pert)).get()
        inp_pert.flat[pert_ind] = save_val - epsilon
        reset_lstm(lstm)
        lstm.allocate()
        out_neg = lstm.fprop(lstm.be.array(inp_pert)).get()
        loss_pos = np.sum(rand_scale * out_pos)
        loss_neg = np.sum(rand_scale * out_neg)
        grad = 0.5 / float(epsilon) * (loss_pos - loss_neg)
        grads_est.flat[pert_ind] = grad
        inp_pert.flat[pert_ind] = save_val
    del lstm
    return (grads_est, deltas_neon)