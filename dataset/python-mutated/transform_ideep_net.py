import argparse
import copy
import json
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, utils
import caffe2.python._import_c_extension as C

def pairwise(iterable):
    if False:
        for i in range(10):
            print('nop')
    from itertools import tee
    (a, b) = tee(iterable)
    next(b, None)
    return zip(a, b)

def last_producer(ops, blob):
    if False:
        for i in range(10):
            print('nop')
    for (i, op) in reversed(list(enumerate(ops))):
        if blob in op.output:
            return i
    raise ValueError('Failed to find last producer of blob, %s', blob)

def blob_uses(net, blob):
    if False:
        print('Hello World!')
    u = []
    for (i, op) in enumerate(net.op):
        if blob in op.input or blob in op.control_input:
            u.append(i)
    return u

def GetArgumentParser():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Caffe2 optimization')
    parser.add_argument('--init_net', type=argparse.FileType('rb'), help='init net')
    parser.add_argument('--pred_net', type=argparse.FileType('rb'), help='predict net')
    parser.add_argument('--verify_input', type=argparse.FileType('r'), help='input dims for verification')
    parser.add_argument('--fuse_bn', default=False, action='store_true')
    parser.add_argument('--fuse_mul_add', default=False, action='store_true')
    parser.add_argument('--fuse_conv_relu', default=False, action='store_true')
    return parser

def fuse_first_bn(net, params, removed_tensors):
    if False:
        while True:
            i = 10
    net = copy.deepcopy(net)
    params = copy.deepcopy(params)
    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue
        if current.type not in ('Conv', 'ConvTranspose') or next_.type != 'SpatialBN':
            continue
        if len(blob_uses(net, current.output[0])) != 1:
            continue
        conv = current
        bn = next_
        fused_conv = copy.deepcopy(conv)
        fused_conv.output[0] = bn.output[0]
        if len(fused_conv.input) != 3:
            bias_name = '{}_bias'.format(conv.input[1])
            net.external_input.extend([bias_name])
            fused_conv.input.extend([bias_name])
            for arg in fused_conv.arg:
                if arg.name == 'no_bias':
                    arg.i = 0
        conv_weight = params[conv.input[1]]
        conv_bias = params[conv.input[2]] if len(conv.input) == 3 else np.zeros(shape=conv_weight.shape[0]).astype(np.float32)
        bn_scale = params[bn.input[1]]
        bn_bias = params[bn.input[2]]
        bn_running_mean = params[bn.input[3]]
        bn_running_var = params[bn.input[4]]
        eps = 1e-05
        for arg in bn.arg:
            if arg.name == 'epsilon':
                eps = arg.f
        A = bn_scale * 1.0 / np.sqrt(bn_running_var + eps)
        B = bn_bias - bn_running_mean * A
        A_ = A.reshape(-1, 1, 1, 1) if conv.type == 'Conv' else A.reshape(1, -1, 1, 1)
        C = conv_bias * A + B
        Q = conv_weight * A_
        params[fused_conv.input[1]] = Q
        params[fused_conv.input[2]] = C
        new_ops = net.op[:i] + [fused_conv] + net.op[j + 1:]
        del net.op[:]
        removed_tensors.append(bn.input[1])
        removed_tensors.append(bn.input[2])
        removed_tensors.append(bn.input[3])
        removed_tensors.append(bn.input[4])
        del params[bn.input[1]]
        del params[bn.input[2]]
        del params[bn.input[3]]
        del params[bn.input[4]]
        net.op.extend(new_ops)
        break
    return (net, params, removed_tensors)

def fuse_bn(net, params, ignore_failure):
    if False:
        for i in range(10):
            print('nop')
    removed_tensors = []
    while True:
        (next_net, next_params, removed_tensors) = fuse_first_bn(net, params, removed_tensors)
        if len(next_net.op) == len(net.op):
            if any((op.type == 'SpatialBN' for op in next_net.op)) and (not ignore_failure):
                raise Exception('Model contains SpatialBN op after fusion: %s', next_net)
            return (next_net, next_params, removed_tensors)
        (net, params, removed_tensors) = (next_net, next_params, removed_tensors)

def fuse_first_mul_add(net, params, removed_tensors):
    if False:
        print('Hello World!')
    net = copy.deepcopy(net)
    params = copy.deepcopy(params)
    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if current.type != 'Mul' or next_.type != 'Add':
            continue
        if next_.input[0] != current.output[0]:
            raise Exception('Failure to fuse')
        if len(blob_uses(net, current.output[0])) != 1:
            raise Exception('Failure to fuse')
        log.info('Fusing at index %s', i)
        mul_ = current
        add_ = next_
        batch_norm = copy.deepcopy(mul_)
        batch_norm.type = 'SpatialBN'
        batch_norm.arg.extend([utils.MakeArgument('is_test', 1)])
        batch_norm.arg.extend([utils.MakeArgument('epsilon', float(1e-09))])

        def s(x):
            if False:
                for i in range(10):
                    print('nop')
            return '{}{}'.format(add_.output[0], x)
        fake_mean = s('_mean')
        fake_var = s('_var')
        del batch_norm.input[:]
        batch_norm.input.extend([mul_.input[0], mul_.input[1], add_.input[1], fake_mean, fake_var])
        params[fake_mean] = np.zeros_like(params[mul_.input[1]])
        params[fake_var] = np.ones_like(params[mul_.input[1]])
        net.external_input.extend([fake_mean, fake_var])
        batch_norm.output[0] = add_.output[0]
        new_ops = net.op[:i] + [batch_norm] + net.op[j + 1:]
        del net.op[:]
        net.op.extend(new_ops)
        break
    return (net, params, removed_tensors)

def fuse_mul_add(net, params):
    if False:
        while True:
            i = 10
    removed_tensors = []
    while True:
        (next_net, next_params, removed_tensors) = fuse_first_mul_add(net, params, removed_tensors)
        if len(next_net.op) == len(net.op):
            return (next_net, next_params, removed_tensors)
        (net, params, removed_tensors) = (next_net, next_params, removed_tensors)

def add_tensor(net, name, blob):
    if False:
        for i in range(10):
            print('nop')
    " Create an operator to store the tensor 'blob',\n        run the operator to put the blob to workspace.\n        uint8 is stored as an array of string with one element.\n    "
    kTypeNameMapper = {np.dtype('float32'): 'GivenTensorFill', np.dtype('int32'): 'GivenTensorIntFill', np.dtype('int64'): 'GivenTensorInt64Fill', np.dtype('uint8'): 'GivenTensorStringFill'}
    shape = blob.shape
    values = blob
    if blob.dtype == np.dtype('uint8'):
        shape = [1]
        values = [str(blob.data)]
    op = core.CreateOperator(kTypeNameMapper[blob.dtype], [], [name], arg=[utils.MakeArgument('shape', shape), utils.MakeArgument('values', values)])
    net.op.extend([op])

def gen_init_net_from_blobs(blobs):
    if False:
        while True:
            i = 10
    ' Generate an initialization net based on a blob dict '
    ret = caffe2_pb2.NetDef()
    for (name, blob) in blobs.items():
        add_tensor(ret, name, blob)
    return ret

def fuse_conv_relu(net):
    if False:
        print('Hello World!')
    net = copy.deepcopy(net)
    device_option = core.DeviceOption(caffe2_pb2.IDEEP)
    for op in net.op:
        op.device_option.CopyFrom(device_option)
    new_net = caffe2_pb2.NetDef()
    new_net.ParseFromString(C.transform_optimizeForMKLDNN(net.SerializeToString()))
    return new_net

def Optimize(args):
    if False:
        while True:
            i = 10
    init_net = caffe2_pb2.NetDef()
    predict_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(args.init_net.read())
    predict_net.ParseFromString(args.pred_net.read())
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    param_dict = {p: workspace.FetchBlob(p) for p in workspace.Blobs()}
    external_inputs = {}
    external_outputs = {}
    if args.verify_input:
        value_info = json.load(args.verify_input)
        input_shapes = {k: v[-1] for (k, v) in value_info.items()}
        print('input info: {}'.format(input_shapes))
        for (k, v) in input_shapes.items():
            external_inputs[k] = np.random.randn(*v).astype(np.float32)
            workspace.FeedBlob(k, external_inputs[k])
        workspace.RunNetOnce(predict_net)
        for o in predict_net.external_output:
            external_outputs[o] = workspace.FetchBlob(o)
    if args.fuse_mul_add:
        (predict_net, param_dict, _) = fuse_mul_add(predict_net, param_dict)
    if args.fuse_bn:
        (predict_net, param_dict, _) = fuse_bn(predict_net, param_dict, False)
    if args.fuse_conv_relu:
        predict_net = fuse_conv_relu(predict_net)
    external_outputs_opt = {}
    if args.verify_input:
        workspace.ResetWorkspace()
        device_option = core.DeviceOption(caffe2_pb2.IDEEP) if args.fuse_conv_relu else core.DeviceOption(caffe2_pb2.CPU)
        with core.DeviceScope(device_option):
            for (k, v) in param_dict.items():
                workspace.FeedBlob(k, v, device_option)
            for (k, v) in external_inputs.items():
                workspace.FeedBlob(k, v, device_option)
            workspace.RunNetOnce(predict_net)
            for o in predict_net.external_output:
                external_outputs_opt[o] = workspace.FetchBlob(o)
                assert np.allclose(external_outputs[o], external_outputs_opt[o], atol=0.001, rtol=0.001)
    for (i, o) in enumerate(predict_net.op):
        print('op[{}]: {}'.format(i, o.type))
    init_net = gen_init_net_from_blobs(param_dict)
    with open('init_net.pb', 'wb') as f:
        f.write(init_net.SerializeToString())
    with open('predict_net.pb', 'wb') as f:
        f.write(predict_net.SerializeToString())
if __name__ == '__main__':
    args = GetArgumentParser().parse_args()
    Optimize(args)