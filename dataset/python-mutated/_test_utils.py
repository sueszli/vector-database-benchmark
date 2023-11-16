from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _
from __future__ import unicode_literals as _
import numpy as np
import numpy.testing as npt
import numpy.random as npr
from onnx import helper, ModelProto, ValueInfoProto, TensorProto, NodeProto
from typing import Any, Sequence, Text, Tuple, Optional, Dict, List, TypeVar
from coremltools.converters.onnx import convert
from coremltools.converters.onnx._converter import SupportedVersion
from coremltools._deps import _IS_MACOS
import sys
'\n   dynamically generate random inputs,\n   use caffe2 backend for onnx and\n'

def _forward_onnx_model(model, input_dict, test_name=''):
    if False:
        while True:
            i = 10
    import caffe2.python.onnx.backend
    prepared_backend = caffe2.python.onnx.backend.prepare(model)
    out = prepared_backend.run(input_dict)
    out_dict = {}
    out_names = [v.name for v in model.graph.output]
    for out_name in out_names:
        out_dict[out_name] = out[out_name]
    result = [out[v.name] for v in model.graph.output]
    output_shapes = [_shape_from_onnx_value_info(o) for o in model.graph.output]
    for (i, output) in enumerate(result):
        result[i] = output.reshape(output_shapes[i])
    return np.array(result)

def _onnx_create_model(nodes, inputs, outputs, initializer=[]):
    if False:
        return 10
    initializer_inputs = [helper.make_tensor_value_info(t.name, TensorProto.FLOAT, t.dims) for t in initializer]
    graph = helper.make_graph(nodes=nodes, name='test', inputs=initializer_inputs + [helper.make_tensor_value_info(input_[0], TensorProto.FLOAT, input_[1]) for input_ in inputs], outputs=[helper.make_tensor_value_info(output_[0], output_[2], output_[1]) for output_ in outputs], initializer=initializer)
    onnx_model = helper.make_model(graph)
    return onnx_model

def _onnx_create_single_node_model(op_type, input_shapes, output_shapes, initializer=[], **kwargs):
    if False:
        for i in range(10):
            print('nop')
    inputs = [('input{}'.format(i), input_shapes[i]) for i in range(len(input_shapes))]
    outputs = [('output{}'.format(i), output_shapes[i], TensorProto.FLOAT) for i in range(len(output_shapes))]
    node = helper.make_node(op_type, inputs=[i[0] for i in inputs] + [t.name for t in initializer], outputs=[o[0] for o in outputs], **kwargs)
    return _onnx_create_model([node], inputs, outputs, initializer)

def _shape_from_onnx_value_info(v):
    if False:
        for i in range(10):
            print('nop')
    return tuple([d.dim_value for d in v.type.tensor_type.shape.dim])

def _coreml_forward_model(model, input_dict, output_names, minimum_ios_deployment_target='12'):
    if False:
        print('Hello World!')
    if not SupportedVersion.is_nd_array_supported(minimum_ios_deployment_target):
        for (k, arr) in input_dict.items():
            if len(arr.shape) == 4:
                input_dict[k] = arr[0]
        for (k, v) in input_dict.items():
            if len(v.shape) == 2 and v.shape[0] == 1:
                input_dict[k] = v.flatten()
    coreml_out = model.predict(input_dict, useCPUOnly=True)
    return np.array([coreml_out[name] for name in output_names])

def _coreml_forward_onnx_model(model, input_dict, onnx_coreml_input_shape_map={}, minimum_ios_deployment_target='12'):
    if False:
        return 10
    coreml_model = convert(model, onnx_coreml_input_shape_map=onnx_coreml_input_shape_map, minimum_ios_deployment_target=minimum_ios_deployment_target)
    output_names = [o.name for o in model.graph.output]
    return _coreml_forward_model(coreml_model, input_dict, output_names, minimum_ios_deployment_target=minimum_ios_deployment_target)

def _random_array(shape, random_seed=10):
    if False:
        i = 10
        return i + 15
    if random_seed:
        npr.seed(random_seed)
    return npr.ranf(shape).astype('float32')

def _conv_pool_output_size(input_shape, dilations, kernel_shape, pads, strides):
    if False:
        i = 10
        return i + 15
    output_height = (input_shape[2] + pads[0] + pads[2] - (dilations[0] * (kernel_shape[0] - 1) + 1)) / strides[0] + 1
    output_width = (input_shape[3] + pads[1] + pads[3] - (dilations[1] * (kernel_shape[1] - 1) + 1)) / strides[1] + 1
    return (int(output_height), int(output_width))
_T = TypeVar('_T')

def _assert_outputs(output1, output2, decimal=7):
    if False:
        for i in range(10):
            print('nop')
    npt.assert_equal(len(output1), len(output2))
    for (o1, o2) in zip(output1, output2):
        npt.assert_almost_equal(o2.flatten(), o1.flatten(), decimal=decimal)

def _prepare_inputs_for_onnx(model, test_name='', values=None):
    if False:
        return 10
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    input_names = [i.name for i in graph.input if i.name not in initializer_names]
    input_shapes = [tuple([d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.input if i.name not in initializer_names]
    if values is None:
        inputs = [_random_array(shape) for shape in input_shapes]
    else:
        inputs = values
    input_dict = dict(zip(input_names, inputs))
    return input_dict

def _test_onnx_model(model, test_name='', decimal=5, onnx_coreml_input_shape_map={}, coreml_input_shape={}, minimum_ios_deployment_target='12'):
    if False:
        i = 10
        return i + 15
    if not test_name:
        test_name = sys._getframe(1).f_code.co_name
    W = _prepare_inputs_for_onnx(model, test_name=test_name)
    c2_outputs = _forward_onnx_model(model, W, test_name=test_name)
    coreml_input_dict = dict()
    supported_ios_version = ['11.2', '12', '13']
    IOS_13_VERSION = supported_ios_version.index('13')
    for (key, value) in W.items():
        if supported_ios_version.index(minimum_ios_deployment_target) < IOS_13_VERSION and key in coreml_input_shape:
            coreml_input_dict[key] = np.reshape(value, coreml_input_shape[key])
        else:
            coreml_input_dict[key] = value
    if _IS_MACOS:
        coreml_outputs = _coreml_forward_onnx_model(model, coreml_input_dict, onnx_coreml_input_shape_map=onnx_coreml_input_shape_map, minimum_ios_deployment_target=minimum_ios_deployment_target)
        _assert_outputs(c2_outputs, coreml_outputs, decimal=decimal)

def _test_single_node(op_type, input_shapes, output_shapes, initializer=[], decimal=5, test_name='', onnx_coreml_input_shape_map={}, coreml_input_shape={}, minimum_ios_deployment_target='12', **kwargs):
    if False:
        while True:
            i = 10
    model = _onnx_create_single_node_model(op_type, input_shapes, output_shapes, initializer, **kwargs)
    if not test_name:
        test_name = sys._getframe(1).f_code.co_name
    _test_onnx_model(model, test_name=test_name, decimal=decimal, onnx_coreml_input_shape_map=onnx_coreml_input_shape_map, coreml_input_shape=coreml_input_shape, minimum_ios_deployment_target=minimum_ios_deployment_target)