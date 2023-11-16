"""
Neural Network optimization utilities.
"""
import numpy as _np

def _fuse_layer_with_scale_layer(layer_idx, scale_idx, layers):
    if False:
        while True:
            i = 10
    layer_type = layers[layer_idx].WhichOneof('layer')
    if layer_type == 'convolution':
        layer = layers[layer_idx].convolution
    elif layer_type == 'innerProduct':
        layer = layers[layer_idx].innerProduct
    else:
        raise Exception('Scale fusion not supper for layer type {} '.format(layer_type))
    scale = layers[scale_idx].scale
    sw = _np.array(scale.scale.floatValue)
    w = _np.array(layer.weights.floatValue)
    w = w.reshape(layer.outputChannels, int(len(w) / layer.outputChannels))
    wp = w * sw[:, None]
    del layer.weights.floatValue[:]
    layer.weights.floatValue.extend(wp.flatten())
    if scale.hasBias:
        sb = _np.array(scale.bias.floatValue)
        if not layer.hasBias:
            layer.bias.floatValue.extend(sb)
            layer.hasBias = True
        else:
            lb = _np.array(layer.bias.floatValue)
            bp = sw * lb + sb
            del layer.bias.floatValue[:]
            layer.bias.floatValue.extend(bp)
    print('Fused {}->{}'.format(layers[layer_idx].name, layers[scale_idx].name))
    del layers[layer_idx].output[:]
    layers[layer_idx].output.extend(layers[scale_idx].output)
    del layers[scale_idx]

def _fuse_layer_with_bias_layer(layer_idx, bias_idx, layers):
    if False:
        while True:
            i = 10
    layer_type = layers[layer_idx].WhichOneof('layer')
    if layer_type == 'convolution':
        layer = layers[layer_idx].convolution
    elif layer_type == 'innerProduct':
        layer = layers[layer_idx].innerProduct
    else:
        raise Exception('Bias fusion not supper for layer type {} '.format(layer_type))
    bias = layers[bias_idx].bias
    bb = _np.array(bias.bias.floatValue)
    if not layer.hasBias:
        layer.bias.floatValue.extend(bb)
        layer.hasBias = True
    else:
        lb = _np.array(layer.bias.floatValue)
        bp = lb + bb
        del layer.bias.floatValue[:]
        layer.bias.floatValue.extend(bp)
    print('Fused {}->{}'.format(layers[layer_idx].name, layers[bias_idx].name))
    del layers[layer_idx].output[:]
    layers[layer_idx].output.extend(layers[bias_idx].output)
    del layers[bias_idx]

def _bn_scale_fusion(bn_idx, scale_idx, layers):
    if False:
        for i in range(10):
            print('nop')
    bn = layers[bn_idx].batchnorm
    scale = layers[scale_idx].scale
    gamma = _np.array(bn.gamma.floatValue)
    beta = _np.array(bn.beta.floatValue)
    sw = _np.array(scale.scale.floatValue)
    gamma = gamma * sw
    beta = beta * sw
    if scale.hasBias:
        sb = _np.array(scale.bias.floatValue)
        beta = beta + sb
    del bn.gamma.floatValue[:]
    del bn.beta.floatValue[:]
    bn.gamma.floatValue.extend(gamma)
    bn.beta.floatValue.extend(beta)
    print('Fused {}->{}'.format(layers[bn_idx].name, layers[scale_idx].name))
    del layers[bn_idx].output[:]
    layers[bn_idx].output.extend(layers[scale_idx].output)
    del layers[scale_idx]

def _conv_bn_fusion(conv_idx, bn_idx, layers):
    if False:
        print('Hello World!')
    conv = layers[conv_idx].convolution
    bn = layers[bn_idx].batchnorm
    mean = _np.array(bn.mean.floatValue)
    variance = _np.array(bn.variance.floatValue) + bn.epsilon
    gamma = _np.array(bn.gamma.floatValue)
    beta = _np.array(bn.beta.floatValue)
    w = _np.array(conv.weights.floatValue)
    if conv.hasBias:
        b = _np.array(conv.bias.floatValue)
    else:
        b = _np.zeros(conv.outputChannels)
    w = w.reshape(conv.outputChannels, int(len(w) / conv.outputChannels))
    wp = (gamma / _np.sqrt(variance))[:, None] * w
    bp = gamma * b / _np.sqrt(variance) - gamma * mean / _np.sqrt(variance) + beta
    del conv.weights.floatValue[:]
    if conv.hasBias:
        del conv.bias.floatValue[:]
    conv.weights.floatValue.extend(wp.flatten())
    conv.bias.floatValue.extend(bp)
    conv.hasBias = True
    print('Fused {}->{}'.format(layers[conv_idx].name, layers[bn_idx].name))
    del layers[conv_idx].output[:]
    layers[conv_idx].output.extend(layers[bn_idx].output)
    del layers[bn_idx]

def _get_nn_mappings(layers):
    if False:
        print('Hello World!')
    layer_map = {}
    type_map = {}
    output_map = {}
    input_map = {}
    for (idx, layer) in enumerate(layers):
        layer_name = '{}'.format(idx)
        layer_map[layer_name] = {'outputs': [], 'inputs': []}
        layer_type = layer.WhichOneof('layer')
        if layer_type not in type_map.keys():
            type_map[layer_type] = []
        type_map[layer_type].append(layer_name)
        for o in layer.output:
            layer_map[layer_name]['outputs'].append(o)
        for i in layer.input:
            layer_map[layer_name]['inputs'].append(i)
    for l in layer_map.keys():
        output_map[l] = []
        input_map[l] = []
        for cl in layer_map.keys():
            if any((x in layer_map[l]['outputs'] for x in layer_map[cl]['inputs'])):
                output_map[l].append(cl)
            if any((x in layer_map[l]['inputs'] for x in layer_map[cl]['outputs'])):
                input_map[l].append(cl)
    return (type_map, output_map, input_map)

def _optimize_nn(layers):
    if False:
        print('Hello World!')
    (type_map, output_map, input_map) = _get_nn_mappings(layers)
    bn_layers = []
    conv_layers = []
    ip_layers = []
    bias_layers = []
    scale_layers = []
    if 'batchnorm' in type_map.keys():
        for bn_layer_idx in type_map['batchnorm']:
            if not layers[int(bn_layer_idx)].batchnorm.instanceNormalization:
                bn_layers.append(bn_layer_idx)
    if 'convolution' in type_map.keys():
        conv_layers = type_map['convolution']
    if 'innerProduct' in type_map.keys():
        ip_layers = type_map['innerProduct']
    if 'bias' in type_map.keys():
        bias_layers = type_map['bias']
    if 'scale' in type_map.keys():
        scale_layers = type_map['scale']
    for conv_idx in conv_layers:
        if len(output_map[conv_idx]) != 1:
            continue
        output_idx = output_map[conv_idx][0]
        if len(input_map[output_idx]) != 1:
            continue
        if output_idx in bn_layers:
            _conv_bn_fusion(int(conv_idx), int(output_idx), layers)
            return _optimize_nn(layers)
        if output_idx in scale_layers:
            _fuse_layer_with_scale_layer(int(conv_idx), int(output_idx), layers)
            return _optimize_nn(layers)
        if output_idx in bias_layers:
            _fuse_layer_with_bias_layer(int(conv_idx), int(output_idx), layers)
            return _optimize_nn(layers)
    for ip_idx in ip_layers:
        if len(output_map[ip_idx]) != 1:
            continue
        output_idx = output_map[ip_idx][0]
        if len(input_map[output_idx]) != 1:
            continue
        if output_idx in scale_layers:
            _fuse_layer_with_scale_layer(int(ip_idx), int(output_idx), layers)
            return _optimize_nn(layers)
        if output_idx in bias_layers:
            _fuse_layer_with_bias_layer(int(ip_idx), int(output_idx), layers)
            return _optimize_nn(layers)
    for bn_idx in bn_layers:
        if len(output_map[bn_idx]) != 1:
            continue
        output_idx = output_map[bn_idx][0]
        if len(input_map[output_idx]) != 1:
            continue
        if output_idx in scale_layers:
            _bn_scale_fusion(int(bn_idx), int(output_idx), layers)
            return _optimize_nn(layers)