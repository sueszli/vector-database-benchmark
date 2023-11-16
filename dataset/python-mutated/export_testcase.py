import os
import warnings
import chainer
from onnx_chainer.export import _available
if _available:
    from onnx_chainer.export import export
    from onnx_chainer.onnx_helper import cleanse_param_name
    from onnx_chainer.onnx_helper import write_tensor_pb

def export_testcase(model, args, out_dir, output_grad=False, **kwargs):
    if False:
        print('Hello World!')
    'Export model and I/O tensors of the model in protobuf format.\n\n    Similar to the `export` function, this function first performs a forward\n    computation to a given input for obtaining an output. Then, this function\n    saves the pair of input and output in Protobuf format, which is a\n    defacto-standard format in ONNX.\n\n    This function also saves the model with the name "model.onnx".\n\n    Args:\n        model (~chainer.Chain): The model object.\n        args (list): The arguments which are given to the model\n            directly. Unlike `export` function, only `list` type is accepted.\n        out_dir (str): The directory name used for saving the input and output.\n        output_grad (bool): If True, this function will output model\'s\n            gradient with names \'gradient_%d.pb\'.\n        **kwargs (dict): keyword arguments for ``onnx_chainer.export``.\n    '
    os.makedirs(out_dir, exist_ok=True)
    model.cleargrads()
    (onnx_model, inputs, outputs) = export(model, args, filename=os.path.join(out_dir, 'model.onnx'), return_named_inout=True, no_testcase=True, **kwargs)
    test_data_dir = os.path.join(out_dir, 'test_data_set_0')
    os.makedirs(test_data_dir, exist_ok=True)
    for (i, (name, var)) in enumerate(inputs.items()):
        pb_name = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        array = chainer.cuda.to_cpu(var.array)
        write_tensor_pb(pb_name, name, array)
    for (i, (name, var)) in enumerate(outputs.items()):
        pb_name = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        array = chainer.cuda.to_cpu(var.array)
        write_tensor_pb(pb_name, name, array)
    if output_grad:
        if len(outputs) > 1:
            outputs = chainer.functions.identity(*outputs)
        for out in outputs.values():
            out.grad = model.xp.ones_like(out.array)
        list(outputs.values())[0].backward()
        for (i, (name, param)) in enumerate(model.namedparams()):
            pb_name = os.path.join(test_data_dir, 'gradient_{}.pb'.format(i))
            grad = chainer.cuda.to_cpu(param.grad)
            onnx_name = cleanse_param_name(name)
            if grad is None:
                warnings.warn('Parameter `{}` does not have gradient value'.format(name))
            else:
                write_tensor_pb(pb_name, onnx_name, grad)