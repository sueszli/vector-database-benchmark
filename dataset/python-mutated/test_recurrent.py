"""
This test compares the NEON recurrent layer against a numpy reference recurrent
implementation and compares the NEON recurrent bprop deltas to the gradients
estimated by finite differences.
The numpy reference recurrent layer contains static methods for forward pass
and backward pass.
The test runs a SINGLE layer of recurrent layer and compare numerical values
The reference model handles batch_size as 1 only

The following are made sure to be the same in both recurrent layers
    -   initial h values (all zeros)
    -   initial W, b (ones or random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside recurrent_ref is seq_len, input_size, 1
    -   the data shape inside recurrent (neon) is feature, seq_len * batch_size
"""
import os
import pytest
import subprocess as subp
import itertools as itt
import numpy as np
from neon import NervanaObject, logger as neon_logger
from neon.initializers.initializer import Constant, Gaussian
from neon.layers import Recurrent
from neon.layers.container import DeltasTree
from neon.transforms import Tanh
from recurrent_ref import Recurrent as RefRecurrent
from utils import allclose_with_out
try:
    from neon.backends.nervanacpu import NervanaCPU
except ImportError:

    class NervanaCPU(object):
        pass

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    bsz_rng = [1]
    if 'refgruargs' in metafunc.fixturenames:
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
        metafunc.parametrize('refgruargs', fargs)
    if 'gradgruargs' in metafunc.fixturenames:
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
        metafunc.parametrize('gradgruargs', fargs)

def test_ref_compare_ones(backend_default, refgruargs):
    if False:
        while True:
            i = 10
    (seq_len, input_size, hidden_size, batch_size) = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    check_rnn(seq_len, input_size, hidden_size, batch_size, Constant(val=1.0), [1.0, 0.0])

def test_ref_compare_rand(backend_default, refgruargs):
    if False:
        while True:
            i = 10
    (seq_len, input_size, hidden_size, batch_size) = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    try:
        check_rnn(seq_len, input_size, hidden_size, batch_size, Gaussian())
    except Exception:
        if not isinstance(NervanaObject.be, NervanaCPU):
            check_rnn(seq_len, input_size, hidden_size, batch_size, Gaussian())
        else:
            if os.getenv('PLATFORM'):
                platform = os.getenv('PLATFORM')
            elif os.path.exists('/proc/cpuinfo'):
                cat_cmd = 'cat /proc/cpuinfo | grep "model name" | tail -1 | cut -f 2 -d \':\' |                            cut -f 3 -d \')\' | cut -f 1 -d \'@\' | cut -f 2,3 -d \' \''
                cpu_model_name = subp.check_output(cat_cmd, shell=True)
            else:
                cpu_model_name = 'unknown'
            if cpu_model_name == 'CPU E5-2699A\n' or b'CPU E5-2699A\n':
                platform = 'BDW'
            else:
                platform = 'unknown'
            if platform == 'BDW':
                pytest.xfail(reason='xfail issue #1041 with {} PLATFORM'.format(platform))
            else:
                check_rnn(seq_len, input_size, hidden_size, batch_size, Gaussian())

def check_rnn(seq_len, input_size, hidden_size, batch_size, init_func, inp_moms=[0.0, 1.0]):
    if False:
        return 10
    input_shape = (input_size, seq_len * batch_size)
    output_shape = (hidden_size, seq_len * batch_size)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    rnn = Recurrent(hidden_size, init_func, activation=Tanh())
    rnn_ref = RefRecurrent(input_size, hidden_size)
    Wxh = rnn_ref.Wxh
    Whh = rnn_ref.Whh
    bh = rnn_ref.bh
    inp = np.random.rand(*input_shape) * inp_moms[1] + inp_moms[0]
    inpa = rnn.be.array(inp)
    deltas = np.random.randn(*output_shape)
    inp_ref = inp.copy().T.reshape(seq_len, batch_size, input_size).swapaxes(1, 2)
    deltas_ref = deltas.copy().T.reshape(seq_len, batch_size, hidden_size).swapaxes(1, 2)
    rnn.configure((input_size, seq_len))
    rnn.prev_layer = True
    rnn.allocate()
    dtree = DeltasTree()
    rnn.allocate_deltas(dtree)
    dtree.allocate_buffers()
    rnn.set_deltas(dtree)
    rnn.fprop(inpa)
    Wxh[:] = rnn.W_input.get()
    Whh[:] = rnn.W_recur.get()
    bh[:] = rnn.b.get()
    (dWxh_ref, dWhh_ref, db_ref, h_ref_list, dh_ref_list, d_out_ref) = rnn_ref.lossFun(inp_ref, deltas_ref)
    rnn.bprop(rnn.be.array(deltas))
    dWxh_neon = rnn.dW_input.get()
    dWhh_neon = rnn.dW_recur.get()
    db_neon = rnn.db.get()
    neon_logger.display('====Verifying hidden states====')
    assert allclose_with_out(rnn.outputs.get(), h_ref_list, rtol=0.0, atol=1e-05)
    neon_logger.display('fprop is verified')
    neon_logger.display('====Verifying update on W and b ====')
    neon_logger.display('dWxh')
    assert allclose_with_out(dWxh_neon, dWxh_ref, rtol=0.0, atol=1e-05)
    neon_logger.display('dWhh')
    assert allclose_with_out(dWhh_neon, dWhh_ref, rtol=0.0, atol=1e-05)
    neon_logger.display('====Verifying update on bias====')
    neon_logger.display('db')
    assert allclose_with_out(db_neon, db_ref, rtol=0.0, atol=1e-05)
    neon_logger.display('bprop is verified')
    return

def reset_rnn(rnn):
    if False:
        i = 10
        return i + 15
    rnn.x = None
    rnn.xs = None
    rnn.outputs = None
    return

def test_gradient_neon_gru(backend_default, gradgruargs):
    if False:
        print('Hello World!')
    (seq_len, input_size, hidden_size, batch_size) = gradgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size)

def gradient_check(seq_len, input_size, hidden_size, batch_size, threshold=0.001):
    if False:
        i = 10
        return i + 15
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
        allclose_with_out(grad_est, deltas, rtol=0.0, atol=0.0)
        NervanaObject.be.rng_reset()
    neon_logger.display('Worst case error %e with perturbation %e' % (min_max_err, pert_mag))
    neon_logger.display('Threshold %e' % threshold)
    assert min_max_err < threshold

def gradient_calc(seq_len, input_size, hidden_size, batch_size, epsilon=None, rand_scale=None, inp_bl=None):
    if False:
        for i in range(10):
            print('nop')
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    input_shape = (input_size, seq_len * batch_size)
    if inp_bl is None:
        inp_bl = np.random.randn(*input_shape)
    rnn = Recurrent(hidden_size, Gaussian(), activation=Tanh())
    inpa = rnn.be.array(np.copy(inp_bl))
    rnn.configure((input_size, seq_len))
    rnn.prev_layer = True
    rnn.allocate()
    dtree = DeltasTree()
    rnn.allocate_deltas(dtree)
    dtree.allocate_buffers()
    rnn.set_deltas(dtree)
    out_bl = rnn.fprop(inpa).get()
    if rand_scale is None:
        rand_scale = np.random.random(out_bl.shape) * 2.0 - 1.0
    deltas_neon = rnn.bprop(rnn.be.array(np.copy(rand_scale))).get()
    grads_est = np.zeros(inpa.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inpa.size):
        save_val = inp_pert.flat[pert_ind]
        inp_pert.flat[pert_ind] = save_val + epsilon
        reset_rnn(rnn)
        rnn.allocate()
        out_pos = rnn.fprop(rnn.be.array(inp_pert)).get()
        inp_pert.flat[pert_ind] = save_val - epsilon
        reset_rnn(rnn)
        rnn.allocate()
        out_neg = rnn.fprop(rnn.be.array(inp_pert)).get()
        loss_pos = np.sum(rand_scale * out_pos)
        loss_neg = np.sum(rand_scale * out_neg)
        grad = 0.5 * (loss_pos - loss_neg) / epsilon
        grads_est.flat[pert_ind] = grad
        inp_pert.flat[pert_ind] = save_val
    del rnn
    return (grads_est, deltas_neon)
if __name__ == '__main__':
    from neon.backends import gen_backend
    bsz = 1
    be = gen_backend(backend='gpu', batch_size=bsz)
    fargs = (30, 5, 10, bsz)
    test_ref_compare_rand(be, fargs)