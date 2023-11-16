import io
import onnx
import torch.onnx
from caffe2.python.core import BlobReference, Net
from caffe2.python.onnx.backend import Caffe2Backend
_next_idx = 0

class _FakeDict:

    def __init__(self, fn):
        if False:
            for i in range(10):
                print('nop')
        self.fn = fn

    def get(self, name, _):
        if False:
            return 10
        return self.fn(name)

def PyTorchModule(helper, model, sample_arguments, caffe2_inputs, prefix_name=None):
    if False:
        return 10
    '\n    Embed an ONNX-exportable PyTorch Model into a Caffe2 model being built.\n\n    Args:\n        helper (caffe2.python.core.ModelHelder): the model helper where\n            this imported network should be inserted\n        model (torch.nn.Module): the model to be exported\n        sample_arguments (tuple of arguments): the inputs to\n            the model, e.g., such that ``model(*args)`` is a valid\n            invocation of the model.  Any non-Variable arguments will\n            be hard-coded into the exported model; any Variable arguments\n            will become inputs of the exported model, in the order they\n            occur in args.  If args is a Variable, this is equivalent\n            to having called it with a 1-ary tuple of that Variable.\n            (Note: passing keyword arguments to the model is not currently\n            supported.  Give us a shout if you need it.)\n        caffe2_inputs (list of str or caffe2.python.core.BlobReference): the\n           caffe2 Blobs that should be inputs to this network. Must be\n           the same length as sample_arguments\n        prefix_name: prefix name to add to each member of the blob, if None then\n           a fresh prefix pytorch_input_N/ is used\n    Returns:\n        A tuple of caffe2.python.core.BlobReference objects referring to the\n        models outputs, or a single BlobReference when the model returns a single\n        value.\n    '
    if prefix_name is None:
        global _next_idx
        prefix_name = 'pytorch_import_' + str(_next_idx) + '/'
        _next_idx += 1
    f = io.BytesIO()
    torch.onnx.export(model, sample_arguments, f, export_params=True)
    onnx_model = onnx.load(io.BytesIO(f.getvalue()))
    (init_net, predict_net) = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    initialized = {x.name for x in onnx_model.graph.initializer}
    uninitialized_inputs = {x.name: i for (i, x) in enumerate(onnx_model.graph.input) if x.name not in initialized}
    if len(uninitialized_inputs) != len(caffe2_inputs):
        raise ValueError(f'Expected {len(uninitialized_inputs)} inputs but found {len(caffe2_inputs)}')

    def remap_blob_name(name):
        if False:
            while True:
                i = 10
        if name in uninitialized_inputs:
            idx = uninitialized_inputs[name]
            return str(caffe2_inputs[idx])
        return prefix_name + name
    predict_net = Net(predict_net).Clone('anon', _FakeDict(remap_blob_name))
    helper.net.AppendNet(predict_net)
    init_net = Net(init_net).Clone('anon', _FakeDict(remap_blob_name))
    helper.param_init_net.AppendNet(init_net)
    results = tuple((BlobReference(remap_blob_name(x.name), helper.net) for x in onnx_model.graph.output))
    return results