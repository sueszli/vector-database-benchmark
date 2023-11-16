import torch
import torch.utils.dlpack as torch_dlpack
from nvidia.dali import ops
from nvidia.dali.pipeline import Pipeline

class TorchPythonFunction(ops.PythonFunctionBase):
    schema_name = 'TorchPythonFunction'
    ops.register_cpu_op('TorchPythonFunction')
    ops.register_gpu_op('TorchPythonFunction')

    def _torch_stream_wrapper(self, function, *ins):
        if False:
            print('Hello World!')
        with torch.cuda.stream(self.stream):
            out = function(*ins)
        self.stream.synchronize()
        return out

    def torch_wrapper(self, batch_processing, function, device, *args):
        if False:
            i = 10
            return i + 15
        func = function if device == 'cpu' else lambda *ins: self._torch_stream_wrapper(function, *ins)
        if batch_processing:
            return ops.PythonFunction.function_wrapper_batch(func, self.num_outputs, torch.utils.dlpack.from_dlpack, torch.utils.dlpack.to_dlpack, *args)
        else:
            return ops.PythonFunction.function_wrapper_per_sample(func, self.num_outputs, torch_dlpack.from_dlpack, torch_dlpack.to_dlpack, *args)

    def __call__(self, *inputs, **kwargs):
        if False:
            return 10
        pipeline = Pipeline.current()
        if pipeline is None:
            Pipeline._raise_no_current_pipeline('TorchPythonFunction')
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=pipeline.device_id)
        return super(TorchPythonFunction, self).__call__(*inputs, **kwargs)

    def __init__(self, function, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        if False:
            while True:
                i = 10
        self.stream = None
        super(TorchPythonFunction, self).__init__(impl_name='DLTensorPythonFunctionImpl', function=lambda *ins: self.torch_wrapper(batch_processing, function, device, *ins), num_outputs=num_outputs, device=device, batch_processing=batch_processing, **kwargs)