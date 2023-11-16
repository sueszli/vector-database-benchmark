import collections
import numpy as np
from caffe2.python import utils, workspace
from caffe2.quantization.server import dnnlowp_pybind11
from hypothesis import assume

def check_quantized_results_close(outputs, ref=None, symmetric=False, atol_scale=0.53):
    if False:
        for i in range(10):
            print('nop')
    if ref is None:
        ref = outputs[0][0]
    if ref.size == 0:
        return
    ref_min = min(np.min(ref), 0)
    ref_max = max(np.max(ref), 0)
    if symmetric:
        ref_scale = 2 * max(abs(ref_max), abs(ref_min)) / 255
    else:
        ref_scale = (ref_max - ref_min) / 255
    atol = ref_scale * atol_scale
    for o in outputs[1:]:
        np.testing.assert_allclose(o[0], outputs[0][0], atol=atol, rtol=0)

def pairwise(iterable):
    if False:
        return 10
    's -> (s0,s1), (s1,s2), (s2, s3), ...'
    from itertools import tee
    (a, b) = tee(iterable)
    next(b, None)
    return zip(a, b)

def avoid_vpmaddubsw_overflow_fc(batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max):
    if False:
        return 10
    for (i, j) in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
    for (i, j) in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < 1 << 15

def avoid_vpmaddubsw_overflow(strides, pads, kernels, dilations, sizes, input_channels, output_channels, batch_size, X, X_min, X_max, W, W_min, W_max):
    if False:
        for i in range(10):
            print('nop')
    ndim = len(sizes)
    dkernels = tuple((dilations[i] * (kernels[i] - 1) + 1 for i in range(ndim)))
    size_cols = tuple(((sizes[i] + 2 * pads[i] - dkernels[i]) // strides[i] + 1 for i in range(ndim)))
    for out_idx in np.ndindex((batch_size,) + size_cols + (output_channels,)):
        b = out_idx[0]
        oc = out_idx[-1]
        o_spatial = out_idx[1:-1]
        for (filter_idx1, filter_idx2) in pairwise(np.ndindex(kernels + (input_channels,))):
            f0 = filter_idx1[:-1]
            ic0 = filter_idx1[-1]
            f1 = filter_idx2[:-1]
            ic1 = filter_idx2[-1]
            i0s = tuple((strides[i] * o_spatial[i] - pads[i] + dilations[i] * f0[i] for i in range(ndim)))
            i1s = tuple((strides[i] * o_spatial[i] - pads[i] + dilations[i] * f1[i] for i in range(ndim)))
            w0 = W[(oc,) + f0 + (ic0,)] - 128 - W_min
            w1 = W[(oc,) + f1 + (ic1,)] - 128 - W_min
            if all((0 <= i0s[i] < sizes[i] for i in range(ndim))):
                x0 = X[(b,) + i0s + (ic0,)] - X_min
            else:
                x0 = -X_min
            if all((0 <= i1s[i] < sizes[i] for i in range(ndim))):
                x1 = X[(b,) + i1s + (ic1,)] - X_min
            else:
                x1 = -X_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[(oc,) + f1 + (ic1,)] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 >= 1 << 15:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[(oc,) + f1 + (ic1,)] = int(w1_adjusted) + 128 + W_min
    for out_idx in np.ndindex((batch_size,) + size_cols + (output_channels,)):
        b = out_idx[0]
        oc = out_idx[-1]
        o_spatial = out_idx[1:-1]
        for (filter_idx1, filter_idx2) in pairwise(np.ndindex(kernels + (input_channels,))):
            f0 = filter_idx1[:-1]
            ic0 = filter_idx1[-1]
            f1 = filter_idx2[:-1]
            ic1 = filter_idx2[-1]
            i0s = tuple((strides[i] * o_spatial[i] - pads[i] + dilations[i] * f0[i] for i in range(ndim)))
            i1s = tuple((strides[i] * o_spatial[i] - pads[i] + dilations[i] * f1[i] for i in range(ndim)))
            w0 = W[(oc,) + f0 + (ic0,)] - 128 - W_min
            w1 = W[(oc,) + f1 + (ic1,)] - 128 - W_min
            if all((0 <= i0s[i] < sizes[i] for i in range(ndim))):
                x0 = X[(b,) + i0s + (ic0,)] - X_min
            else:
                x0 = -X_min
            if all((0 <= i1s[i] < sizes[i] for i in range(ndim))):
                x1 = X[(b,) + i1s + (ic1,)] - X_min
            else:
                x1 = -X_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < 1 << 15

def generate_convnd_inputs(strides, pads, kernels, dilations, sizes, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization=False, preserve_activation_sparsity=False, preserve_weight_sparsity=False):
    if False:
        while True:
            i = 10
    dim = len(sizes)
    assume(all((len(a) == dim for a in [strides, pads, kernels, dilations])))
    assume(all((sizes[d] >= dilations[d] * (kernels[d] - 1) + 1 for d in range(dim))))
    input_channels = input_channels_per_group * group
    output_channels = output_channels_per_group * group
    depthwise_convolution = input_channels_per_group == 1 and output_channels_per_group == 1
    assert input_channels > 1
    assert output_channels > 1
    X_min = 0 if preserve_activation_sparsity else -77
    X_max = X_min + 255
    X_range = X_max - X_min
    if depthwise_convolution and groupwise_quantization:
        X_range /= 2
    X = np.round(np.random.rand(*(batch_size,) + tuple(sizes) + (input_channels,)) * X_range + X_min)
    X = X.astype(np.float32)
    if batch_size != 0 and depthwise_convolution and groupwise_quantization and (not preserve_activation_sparsity):
        assert X.shape[1] >= 3
        assert all((X.shape[d + 1] >= kernels[d] + 2 for d in range(1, dim)))
        X_sub = X[(0,) * (X.ndim - dim - 1) + (slice(None),) * dim + (0,)]
        X_sub[(1,) + tuple((kernels[d] // 2 + 1 for d in range(1, dim)))] = X_max
        X_sub[[[0, 2]] + [[kernels[d] + 1, 0] for d in range(1, dim)]] = X_min
        for d1 in range(1, dim):
            X_sub[[[1]] + [[kernels[d2] // 2 + 1] for d2 in range(1, d1)] + [[kernels[d1] // 2, kernels[d1] // 2 + 2]] + [[kernels[d2] + 1, 0] for d2 in range(d1 + 1, dim)]] = X_min
    else:
        X[..., 0] = X_min
        if batch_size != 0:
            X[(0,) * (X.ndim - 1) + (1,)] = X_max
    if preserve_weight_sparsity:
        W_min = -128
        W_max = 100
    else:
        W_min = -100
        W_max = W_min + 255
    W = np.round(np.random.rand(*(output_channels,) + tuple(kernels) + (input_channels_per_group,)) * (W_max - W_min) + W_min)
    W = W.astype(np.float32)
    if groupwise_quantization:
        for g in range(group):
            W[(g * output_channels_per_group,) + (0,) * (W.ndim - 1)] = W_min
            if depthwise_convolution:
                W[(g * output_channels_per_group, 1) + (0,) * (W.ndim - 2)] = W_max
            else:
                assert output_channels_per_group > 1
                W[(g * output_channels_per_group + 1,) + (0,) * (W.ndim - 1)] = W_max
            if not preserve_weight_sparsity:
                W[g * output_channels_per_group:(g + 1) * output_channels_per_group,] += g
    else:
        W[(0,) + (0,) * (W.ndim - 1)] = W_min
        W[(1,) + (0,) * (W.ndim - 1)] = W_max
    different_range_per_group = groupwise_quantization and (not preserve_weight_sparsity)
    for g in range(group):
        avoid_vpmaddubsw_overflow(strides, pads, kernels, dilations, sizes, input_channels_per_group, output_channels_per_group, batch_size, X[..., g * input_channels_per_group:(g + 1) * input_channels_per_group], X_min, X_max, W[g * output_channels_per_group:(g + 1) * output_channels_per_group,], W_min + (g if different_range_per_group else 0), W_max + (g if different_range_per_group else 0))
    if order == 'NCHW':
        X = utils.NHWC2NCHW(X)
        W = utils.NHWC2NCHW(W)
    b = np.random.randn(output_channels).astype(np.float32)
    return (X, W, b)

def generate_conv_inputs(stride, pad, kernel, dilation, size, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization=False, preserve_activation_sparsity=False, preserve_weight_sparsity=False):
    if False:
        return 10
    return generate_convnd_inputs((stride,) * 2, (pad,) * 2, (kernel,) * 2, (dilation,) * 2, (size,) * 2, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization, preserve_activation_sparsity, preserve_weight_sparsity)

def run_conv_or_fc(test_case, init_net, net, X, W, b, op_type, engine, order, gc, outputs, scale=None, zero_point=None, x_scale=None, x_zero_point=None):
    if False:
        return 10
    if order:
        Output = collections.namedtuple('Output', ['Y', 'op_type', 'engine', 'order'])
    else:
        Output = collections.namedtuple('Output', ['Y', 'op_type', 'engine'])
    test_case.ws.create_blob('X').feed(X, device_option=gc)
    test_case.ws.create_blob('W').feed(W, device_option=gc)
    test_case.ws.create_blob('b').feed(b, device_option=gc)
    if scale is not None and zero_point is not None:
        with workspace.WorkspaceGuard(test_case.ws):
            dnnlowp_pybind11.CreateInt8QuantParamsBlob('quant_param', float(scale), int(zero_point))
    if x_scale is not None and x_zero_point is not None:
        with workspace.WorkspaceGuard(test_case.ws):
            dnnlowp_pybind11.CreateInt8QuantParamsBlob('X_quant_param', float(x_scale), int(x_zero_point))
    if init_net:
        test_case.ws.run(init_net)
    for i in range(1 if engine == '' else 2):
        test_case.ws.run(net)
        Y = test_case.ws.blobs['Y'].fetch()
        if order:
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))
        else:
            outputs.append(Output(Y=Y, op_type=op_type, engine=engine))
    if engine != '':
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('W', W)
        workspace.FeedBlob('b', b)
        if scale is not None and zero_point is not None:
            dnnlowp_pybind11.CreateInt8QuantParamsBlob('quant_param', float(scale), int(zero_point))
        if x_scale is not None and x_zero_point is not None:
            dnnlowp_pybind11.CreateInt8QuantParamsBlob('X_quant_param', float(x_scale), int(x_zero_point))
        if init_net:
            workspace.RunNetOnce(init_net)
        workspace.CreateNet(net)
        for i in range(2):
            workspace.RunNet(net)
            Y = workspace.FetchBlob('Y')
            if order:
                outputs.append(Output(Y=Y, op_type=op_type, engine=engine, order=order))
            else:
                outputs.append(Output(Y=Y, op_type=op_type, engine=engine))