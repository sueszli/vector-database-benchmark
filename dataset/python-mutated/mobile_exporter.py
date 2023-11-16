from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2
import numpy as np

def add_tensor(net, name, blob):
    if False:
        while True:
            i = 10
    " Create an operator to store the tensor 'blob',\n        run the operator to put the blob to workspace.\n        uint8 is stored as an array of string with one element.\n    "
    kTypeNameMapper = {np.dtype('float32'): 'GivenTensorFill', np.dtype('int32'): 'GivenTensorIntFill', np.dtype('int64'): 'GivenTensorInt64Fill', np.dtype('uint8'): 'GivenTensorByteStringToUInt8Fill', np.dtype('O'): 'GivenTensorStringFill'}
    shape = blob.shape
    values = blob
    if blob.dtype == np.dtype('uint8'):
        shape = blob.shape
        values = [blob.tobytes()]
    if blob.dtype == np.dtype('O'):
        for blob_val in blob:
            assert isinstance(blob_val, bytes)
    op = core.CreateOperator(kTypeNameMapper[blob.dtype], [], [name], arg=[utils.MakeArgument('shape', shape), utils.MakeArgument('values', values)])
    net.op.extend([op])

def Export(workspace, net, params):
    if False:
        for i in range(10):
            print('nop')
    'Returns init_net and predict_net suitable for writing to disk\n       and loading into a Predictor'
    proto = net if isinstance(net, caffe2_pb2.NetDef) else net.Proto()
    predict_net = caffe2_pb2.NetDef()
    predict_net.CopyFrom(proto)
    init_net = caffe2_pb2.NetDef()
    (ssa, blob_versions) = core.get_ssa(net)
    inputs = []
    for (versioned_inputs, _) in ssa:
        inputs += [name for (name, _) in versioned_inputs]
    input_blobs = [blob_name for (blob_name, version) in blob_versions.items() if version == 0 and blob_name not in params]
    output_blobs = [blob_name for (blob_name, version) in blob_versions.items() if version != 0 and blob_name not in inputs]
    for blob_ref in params:
        blob_name = str(blob_ref)
        blob = workspace.FetchBlob(blob_name)
        add_tensor(init_net, blob_name, blob)
    for blob_name in input_blobs:
        init_net.op.extend([core.CreateOperator('GivenTensorFill', [], [blob_name], arg=[utils.MakeArgument('shape', [1, 1]), utils.MakeArgument('values', [0.0])])])
    del predict_net.external_input[:]
    new_external_inputs = input_blobs
    for external_input in proto.external_input:
        if external_input not in new_external_inputs:
            new_external_inputs.append(external_input)
    predict_net.external_input.extend(new_external_inputs)
    del predict_net.external_output[:]
    predict_net.external_output.extend(output_blobs)
    return (init_net, predict_net)