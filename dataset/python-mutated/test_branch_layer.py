"""
Convolution layer tests
"""
from builtins import zip
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers import Sequential, Conv, Pooling, BranchNode, Affine, Tree
from neon.initializers.initializer import Gaussian, Constant
from neon.transforms import Rectlin
from neon import logger as neon_logger
init1 = Gaussian(scale=0.01)
relu = Rectlin()
bias = Constant(0)
common = dict(activation=relu, init=init1, bias=bias)
pool2s1p1 = dict(fshape=2, padding=1, strides=1)

def make_tree(trunk, branch1, branch2, alphas):
    if False:
        i = 10
        return i + 15
    _trunk = [l['layer'](**l['config']) for l in trunk]
    bnode = [BranchNode(name='bnode')]
    _branch1 = [l['layer'](**l['config']) for l in branch1]
    _branch2 = [l['layer'](**l['config']) for l in branch2]
    v1 = Tree([_trunk + bnode + _branch1, bnode + _branch2], alphas)
    _trunkb = [l['layer'](**l['config']) for l in trunk]
    _branch1b = [l['layer'](**l['config']) for l in branch1]
    _branch2b = [l['layer'](**l['config']) for l in branch2]
    return (v1, _trunkb, _branch1b, _branch2b)

@pytest.mark.hasgpu
def test_branch_model_gpu(backend_gpu):
    if False:
        while True:
            i = 10
    be = NervanaObject.be
    trunk = [{'layer': Conv, 'config': dict(fshape=(5, 5, 16), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}]
    branch1 = [{'layer': Conv, 'config': dict(fshape=(5, 5, 32), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}, {'layer': Affine, 'config': dict(nout=200, **common)}, {'layer': Affine, 'config': dict(nout=10, init=init1, activation=relu)}]
    branch2 = [{'layer': Conv, 'config': dict(fshape=(3, 3, 32), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}, {'layer': Affine, 'config': dict(nout=256, **common)}, {'layer': Affine, 'config': dict(nout=10, init=init1, activation=relu)}]
    alphas = [1, 1]
    (neon_layer, t, b1, b2) = make_tree(trunk, branch1, branch2, alphas)
    inshape = (16, 32, 32)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, be.bsz))
    inp = be.array(inpa)
    neon_layer.configure(inshape)
    neon_layer.allocate()
    neon_layer.allocate_deltas()
    neon_out = [i.get() for i in neon_layer.fprop(inp)]
    ref_layers = [Sequential(t), Sequential(b1), Sequential(b2)]
    ref_layers[0].configure(inshape)
    ref_layers[1].configure(ref_layers[0].out_shape)
    ref_layers[2].configure(ref_layers[0].out_shape)
    [r.allocate() for r in ref_layers]
    [r.allocate_deltas() for r in ref_layers]
    ref_all_layers = ref_layers[0].layers + ref_layers[1].layers + ref_layers[2].layers
    ref_weight_layers = [l for l in ref_all_layers if l.has_params]
    neon_weight_layers = neon_layer.layers_to_optimize
    for (rl, nl) in zip(ref_weight_layers, neon_weight_layers):
        rl.set_params({'params': {'W': nl.W.get()}})
    inp_middle = ref_layers[0].fprop(inp)
    ref_out = [r.fprop(inp_middle).get() for r in ref_layers[1:]]
    for (h, r) in zip(neon_out, ref_out):
        difference = np.max(np.abs(h - r))
        assert difference < 1e-09
    erra = [np.random.random(ll.shape) for ll in neon_out]
    err = [be.array(e) for e in erra]
    input_layer = neon_layer.layers[0].layers[0]
    input_layer.prev_layer = True
    input_layer.deltas = be.iobuf(inshape)
    neon_layer.bprop(err)
    errp = input_layer.deltas.get()
    for (i, r) in enumerate(ref_layers):
        r.layers[0].prev_layer = True
        _inshape = inshape if i == 0 else ref_layers[0].out_shape
        r.layers[0].deltas = be.iobuf(_inshape)
    joined_err = be.iobuf(ref_layers[0].out_shape)
    branch_errs = [r.bprop(e, a) for (r, e, a) in reversed(list(zip(ref_layers[1:], err, alphas)))]
    joined_err[:] = branch_errs[0] + branch_errs[1]
    err_ref = ref_layers[0].bprop(joined_err).get()
    difference = np.max(np.abs(err_ref - errp))
    neon_logger.display('Max difference: {}'.format(difference))
    assert difference < 1e-09

@pytest.mark.mkl_only
def test_branch_model_mkl(backend_default_mkl):
    if False:
        i = 10
        return i + 15
    be = NervanaObject.be
    trunk = [{'layer': Conv, 'config': dict(fshape=(5, 5, 16), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}]
    branch1 = [{'layer': Conv, 'config': dict(fshape=(5, 5, 32), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}, {'layer': Affine, 'config': dict(nout=200, **common)}, {'layer': Affine, 'config': dict(nout=10, init=init1, activation=relu)}]
    branch2 = [{'layer': Conv, 'config': dict(fshape=(3, 3, 32), **common)}, {'layer': Pooling, 'config': dict(op='max', **pool2s1p1)}, {'layer': Affine, 'config': dict(nout=256, **common)}, {'layer': Affine, 'config': dict(nout=10, init=init1, activation=relu)}]
    alphas = [1, 1]
    (neon_layer, t, b1, b2) = make_tree(trunk, branch1, branch2, alphas)
    inshape = (16, 32, 32)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, be.bsz))
    inp = be.array(inpa)
    neon_layer.configure(inshape)
    neon_layer.allocate()
    neon_layer.allocate_deltas()
    neon_out = [i.get() for i in neon_layer.fprop(inp)]
    ref_layers = [Sequential(t), Sequential(b1), Sequential(b2)]
    ref_layers[0].configure(inshape)
    ref_layers[1].configure(ref_layers[0].out_shape)
    ref_layers[2].configure(ref_layers[0].out_shape)
    [r.allocate() for r in ref_layers]
    [r.allocate_deltas() for r in ref_layers]
    ref_all_layers = ref_layers[0].layers + ref_layers[1].layers + ref_layers[2].layers
    ref_weight_layers = [l for l in ref_all_layers if l.has_params]
    neon_weight_layers = neon_layer.layers_to_optimize
    for (rl, nl) in zip(ref_weight_layers, neon_weight_layers):
        rl.set_params({'params': {'W': nl.W.get()}})
    inp_middle = ref_layers[0].fprop(inp)
    ref_out = [r.fprop(inp_middle).get() for r in ref_layers[1:]]
    for (h, r) in zip(neon_out, ref_out):
        difference = np.max(np.abs(h - r))
        assert difference < 0.01
    erra = [np.random.random(ll.shape) for ll in neon_out]
    err = [be.array(e) for e in erra]
    input_layer = neon_layer.layers[0].layers[0]
    input_layer.prev_layer = True
    input_layer.deltas = be.iobuf(inshape)
    neon_layer.bprop(err)
    errp = input_layer.deltas.get()
    for (i, r) in enumerate(ref_layers):
        r.layers[0].prev_layer = True
        _inshape = inshape if i == 0 else ref_layers[0].out_shape
        r.layers[0].deltas = be.iobuf(_inshape)
    joined_err = be.iobuf(ref_layers[0].out_shape)
    branch_errs = [r.bprop(e, a) for (r, e, a) in reversed(list(zip(ref_layers[1:], err, alphas)))]
    joined_err[:] = branch_errs[0] + branch_errs[1]
    err_ref = ref_layers[0].bprop(joined_err).get()
    difference = np.max(np.abs(err_ref - errp))
    neon_logger.display('Max difference: {}'.format(difference))
    assert difference < 0.001