"""
Generalized gradient testing applied to different layers and activations
"""
import numpy as np
from neon import logger as neon_logger
from neon.layers.container import DeltasTree

def sweep_epsilon(layer, inp, pert_rng, out_shape=None, lshape=None, pert_frac=0.1):
    if False:
        i = 10
        return i + 15
    if out_shape is None:
        inpa = layer.be.array(inp.copy())
        in_shape = lshape if lshape is not None else inpa.shape[0]
        layer.configure(in_shape)
        out_shape = layer.out_shape
    loss_scale = np.random.random(out_shape) * 2.0 - 1.0
    pert_cnt = int(np.ceil(inpa.size * pert_frac))
    pert_inds = np.random.permutation(inpa.size)[0:pert_cnt]
    layer.be.rng_reset()
    min_max_diff = -1.0
    min_max_pert = None
    neon_logger.display('epsilon, max diff')
    for epsilon in pert_rng:
        (max_abs, max_rel) = general_gradient_comp(layer, inp, epsilon=epsilon, loss_scale=loss_scale, lshape=lshape, pert_inds=pert_inds)
        layer.be.rng_reset()
        if min_max_diff < 0 or max_abs < min_max_diff:
            min_max_diff = max_abs
            min_max_pert = epsilon
        neon_logger.display('%e %e %e' % (epsilon, max_abs, max_rel))
        neon_logger.display('Min max diff : %e at Pert. Mag. %e' % (min_max_diff, min_max_pert))
    return (min_max_pert, min_max_diff)

def general_gradient_comp(layer, inp, epsilon=1e-05, loss_scale=None, lshape=None, pert_inds=None, pooling=False):
    if False:
        while True:
            i = 10
    layer.reset()
    inpa = layer.be.array(inp.copy())
    in_shape = lshape if lshape is not None else inpa.shape[0]
    layer.configure(in_shape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()
    dtree = DeltasTree()
    layer.allocate_deltas(dtree)
    dtree.allocate_buffers()
    layer.set_deltas(dtree)
    out = layer.fprop(inpa).get()
    out_shape = out.shape
    if loss_scale is None:
        loss_scale = np.random.random(out_shape) * 2.0 - 1.0
    bprop_deltas = layer.bprop(layer.be.array(loss_scale.copy())).get()
    max_abs_err = -1.0
    max_rel_err = -1.0
    inp_pert = inp.copy()
    if pert_inds is None:
        pert_inds = list(range(inp.size))
    for pert_ind in pert_inds:
        save_val = inp_pert.flat[pert_ind]
        inp_pert.flat[pert_ind] = save_val + epsilon
        layer.reset()
        layer.configure(in_shape)
        layer.allocate()
        inpa = layer.be.array(inp_pert.copy())
        out_pos = layer.fprop(inpa).get().copy()
        inp_pert.flat[pert_ind] = save_val - epsilon
        inpa = layer.be.array(inp_pert.copy())
        layer.reset()
        layer.configure(in_shape)
        layer.allocate()
        out_neg = layer.fprop(inpa).get().copy()
        loss_pos = np.sum(loss_scale * out_pos)
        loss_neg = np.sum(loss_scale * out_neg)
        grad_est = 0.5 * (loss_pos - loss_neg) / epsilon
        inp_pert.flat[pert_ind] = save_val
        bprop_val = bprop_deltas.flat[pert_ind]
        abs_err = abs(grad_est - bprop_val)
        if abs_err > max_abs_err:
            max_abs_err = abs_err
            max_abs_vals = [grad_est, bprop_val]
        if abs(grad_est) + abs(bprop_val) == 0.0:
            rel_err = 0.0
        else:
            rel_err = float(abs_err) / (abs(grad_est) + abs(bprop_val))
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            max_rel_vals = [grad_est, bprop_val]
    neon_logger.display('Worst case diff %e, vals grad: %e, bprop: %e' % (max_abs_err, max_abs_vals[0], max_abs_vals[1]))
    neon_logger.display('Worst case diff %e, vals grad: %e, bprop: %e' % (max_rel_err, max_rel_vals[0], max_rel_vals[1]))
    return (max_abs_err, max_rel_err)