import operator
import torch
from ..common import compare_version

def jit_convert(model, input_sample, jit_method=None, jit_strict=True, example_kwarg_inputs=None):
    if False:
        return 10
    '\n    Internal function to export pytorch model to TorchScript.\n\n    :param model: the model(nn.module) to be transform\n    :param input_sample: torch.Tensor or a list for the model tracing.\n    :param jit_method: use ``jit.trace`` or ``jit.script`` to convert a model\n        to TorchScript.\n    :param jit_strict: Whether recording your mutable container types.\n    :param example_kwarg_inputs: keyword arguments of example inputs that will be\n        passed to ``torch.jit.trace``. Default to ``None``. Either this argument or\n        ``input_sample`` should be specified when use_jit is ``True`` and torch > 2.0,\n        otherwise will be ignored.\n    '
    if jit_method == 'trace':
        if compare_version('torch', operator.ge, '2.0'):
            model = torch.jit.trace(model, example_inputs=input_sample, check_trace=False, strict=jit_strict, example_kwarg_inputs=example_kwarg_inputs)
        else:
            model = torch.jit.trace(model, input_sample, check_trace=False, strict=jit_strict)
    elif jit_method == 'script':
        model = torch.jit.script(model)
    else:
        try:
            if compare_version('torch', operator.ge, '2.0'):
                model = torch.jit.trace(model, example_inputs=input_sample, check_trace=False, strict=jit_strict, example_kwarg_inputs=example_kwarg_inputs)
            else:
                model = torch.jit.trace(model, input_sample, check_trace=False, strict=jit_strict)
        except Exception:
            model = torch.jit.script(model)
    return model