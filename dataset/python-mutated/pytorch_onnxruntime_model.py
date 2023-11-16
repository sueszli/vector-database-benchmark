import torch
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from ..core.onnxruntime_model import ONNXRuntimeModel
import onnxruntime
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.pytorch import export_to_onnx, get_input_example, get_forward_args
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.context_manager import generate_context_manager, BaseContextManager
from bigdl.nano.utils.pytorch import patch_attrs_from_model_to_object, MetaData
from bigdl.nano.utils.common import SafePickle

class PytorchONNXRuntimeModel(ONNXRuntimeModel, AcceleratedLightningModule):
    """
        This is the accelerated model for pytorch and onnxruntime.
        All the external API is based on Trainer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchONNXRuntimeModel will serve for all precision models.
    """

    def __init__(self, model, input_sample=None, onnxruntime_session_options=None, simplification=True, dynamic_axes=True, output_tensors=True, output_metadata=None, **export_kwargs):
        if False:
            print('Hello World!')
        "\n        Create a ONNX Runtime model from pytorch.\n\n        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference\n                      2. Path to ONNXRuntime saved model.\n        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or\n                             model is a LightningModule with any dataloader attached,\n                             defaults to None.\n        :param onnxruntime_session_options: A session option for onnxruntime accelerator.\n        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when\n                               accelerator='onnxruntime', otherwise will be ignored. If this option\n                               is set to True, new dependency 'onnxsim' need to be installed.\n        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model\n                             will have the first dim of each Tensor input as a dynamic batch_size.\n                             If dynamic_axes=False, the exported model will have the shapes of all\n                             input and output tensors set to exactly match those given in\n                             input_sample. To specify axes of tensors as dynamic (i.e. known only\n                             at run-time), set dynamic_axes to a dict with schema:\n\n                             | KEY (str): an input or output name. Each name must also be provided\n                             | in input_names or output_names.\n                             |\n                             | VALUE (dict or list): If a dict, keys are axis indices and values\n                             | are axis names. If a list, each element is an axis index.\n\n                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.\n        :param output_tensors: boolean, default to True and output of the model will be Tensors.\n                               If output_tensors=False, output of the ONNX model will be ndarray.\n        :param output_metadata: metadata of model output, defaults to None.\n        :param **export_kwargs: will be passed to torch.onnx.export function.\n        "
        self.output_metadata = output_metadata
        with TemporaryDirectory() as tmpdir:
            if isinstance(model, torch.nn.Module):
                onnx_path = os.path.join(tmpdir, 'tmp.onnx')
                export_to_onnx(model, input_sample=input_sample, onnx_path=onnx_path, dynamic_axes=dynamic_axes, **export_kwargs)
                if simplification is True:
                    try:
                        from bigdl.nano.deps.onnxsim.onnxsim_api import onnx_simplify
                        onnx_simplify(onnx_path)
                    except Exception:
                        pass
                with BaseContextManager():
                    forward_args = get_forward_args(model)
                    input_sample = get_input_example(model, input_sample, forward_args)
                    if isinstance(input_sample, (tuple, list)):
                        output = model(*input_sample)
                    else:
                        output = model(input_sample)
                    self.output_metadata = MetaData.construct_matadata(output)
            else:
                onnx_path = model
            AcceleratedLightningModule.__init__(self, None)
            ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)
        if onnxruntime_session_options.intra_op_num_threads > 0:
            self.thread_num = onnxruntime_session_options.intra_op_num_threads
        else:
            self.thread_num = None
        self._nano_context_manager = generate_context_manager(accelerator=None, precision='fp32', thread_num=self.thread_num)
        if isinstance(model, torch.nn.Module):
            patch_attrs_from_model_to_object(model, self)
        self.output_tensors = output_tensors

    def on_forward_start(self, inputs):
        if False:
            i = 10
            return i + 15
        if self.ortsess is None:
            invalidInputError(False, 'Please create an instance by PytorchONNXRuntimeModel()')
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_start_kwargs(self, **kwargs):
        if False:
            print('Hello World!')
        self.cope_with_keyword_arguments(kwargs)
        return kwargs

    def on_forward_end(self, outputs):
        if False:
            for i in range(10):
                print('nop')
        if self.output_tensors:
            outputs = self.numpy_to_tensors(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]
        if self.output_metadata is not None:
            outputs = MetaData.reconstruct_output(outputs, self.output_metadata)
        return outputs

    @property
    def status(self):
        if False:
            return 10
        status = super().status
        status.update({'onnx_path': 'onnx_saved_model.onnx', 'metadata_path': 'matadata.pkl', 'intra_op_num_threads': self.session_options.intra_op_num_threads, 'inter_op_num_threads': self.session_options.inter_op_num_threads, 'output_tensors': self.output_tensors})
        return status

    @staticmethod
    def _load(path):
        if False:
            i = 10
            return i + 15
        '\n        Load an ONNX model for inference from directory.\n\n        :param path: Path to model to be loaded.\n        :return: PytorchONNXRuntimeModel model for ONNX Runtime inference.\n        '
        status = PytorchONNXRuntimeModel._load_status(path)
        if status.get('onnx_path', None):
            onnx_path = Path(status['onnx_path'])
            invalidInputError(onnx_path.suffix == '.onnx', "Path of onnx model must be with '.onnx' suffix.")
        else:
            invalidInputError(False, "nano_model_meta.yml must specify 'onnx_path' for loading.")
        onnx_path = Path(path) / status['onnx_path']
        onnxruntime_session_options = onnxruntime.SessionOptions()
        if status.get('intra_op_num_threads', None):
            onnxruntime_session_options.intra_op_num_threads = status.get('intra_op_num_threads', None)
        if status.get('inter_op_num_threads', None):
            onnxruntime_session_options.inter_op_num_threads = status.get('inter_op_num_threads', None)
        output_tensors = status.get('output_tensors', True)
        metadata_path = status.get('metadata_path', None)
        if metadata_path is None or not metadata_path:
            output_metadata = None
        else:
            with open(path / status['metadata_path'], 'rb') as f:
                output_metadata = SafePickle.load(f)
        return PytorchONNXRuntimeModel(str(onnx_path), onnxruntime_session_options=onnxruntime_session_options, output_tensors=output_tensors, output_metadata=output_metadata)

    def _save_model(self, path, compression='fp32'):
        if False:
            return 10
        onnx_path = Path(path) / self.status['onnx_path']
        super()._save_model(onnx_path)
        with open(path / self.status['metadata_path'], 'wb') as f:
            SafePickle.dump(self.output_metadata, f)