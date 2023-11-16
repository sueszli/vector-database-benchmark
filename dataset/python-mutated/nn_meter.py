from __future__ import annotations
import functools
import logging
import tempfile
from typing import Any, Iterable
import torch
from torch import nn
from nn_meter import load_latency_predictor, nnMeterPredictor
from nni.mutable import Mutable, MutableList, LabeledMutable, Sample, MutableExpression
from nni.nas.nn.pytorch import ModelSpace, MutableModule, LayerChoice, Repeat
from nni.nas.profiler import ExpressionProfiler
from .utils import MutableShape, ShapeTensor, standardize_arguments, submodule_input_output_shapes, is_leaf_module, concat_name
_logger = logging.getLogger(__name__)

class _mutable_module_wrapper(MutableModule):

    def __init__(self, module: nn.Module):
        if False:
            return 10
        super().__init__()
        self.module = module

def to_onnx(model: nn.Module, example_inputs: Any) -> Any:
    if False:
        i = 10
        return i + 15
    'Helper function to convert a model to onnx model.'
    try:
        import onnx
        import onnxsim
        import onnxruntime
    except ImportError:
        _logger.error('Please install onnx, onnxruntime, onnxsim to use this function.')
        raise
    with tempfile.TemporaryFile() as fp:
        torch.onnx.export(model, example_inputs, fp, export_params=False)
        fp.seek(0)
        model = onnx.load(fp, load_external_data=False)
    (model_simp, check) = onnxsim.simplify(model)
    if not check:
        _logger.error(f'Check did not pass when simplifying the module with onnxsim. Trying without simplification: {model}')
        model_simp = model
    return model_simp

def sample_to_condition(mutables: dict[str, LabeledMutable], sample: Sample) -> MutableExpression[bool] | bool:
    if False:
        i = 10
        return i + 15
    'Convert a sample to a condition that can be used to verify whether a new sample is compatible with the old one.\n    Freeze the returned condition with a certain sample to get a boolean value.\n\n    Parameters\n    ----------\n    mutables\n        A dictionary mapping label to mutable. Get it from :meth:`Mutable.simplify`.\n    sample\n        A sample to convert.\n    '
    conditions = [mutables[label] == value for (label, value) in sample.items()]
    return functools.reduce(lambda x, y: x & y, conditions, True)

def combinations(module: Mutable | nn.Module, input_shape: tuple[MutableShape, ...]) -> Iterable[tuple[MutableExpression[bool] | bool, Any, Any]]:
    if False:
        while True:
            i = 10
    'List all the combinations of the (mutable) module and the input shape.\n\n    The returned iterator yields a tuple of (sample, module, input) for each combination.\n    The inputs will be generated with :func:`torch.randn` based on the sampled input shape.\n\n    The module can be potentially not any mutable object.\n    If the module is not a ``Mutable``, it must be a ``nn.Module`` so that it can be wrapped with a MutableModule.\n    '
    from torch.utils._pytree import tree_flatten, tree_unflatten
    if not isinstance(module, Mutable):
        module = _mutable_module_wrapper(module)
    (input_shape_flattened, input_shape_spec) = tree_flatten(input_shape)
    all_mutables = MutableList([module] + input_shape_flattened)
    _logger.debug('Search space for current module: %s', module.simplify())
    _logger.debug('Search space for current input shape: %s', MutableList(input_shape_flattened).simplify())
    sample = {}
    for mutables in all_mutables.grid(memo=sample):
        sampled_module = mutables[0]
        if isinstance(sampled_module, _mutable_module_wrapper):
            sampled_module = sampled_module.module
        example_inputs_flattened = [torch.randn(*shape) if isinstance(shape, torch.Size) else shape for shape in mutables[1:]]
        sampled_input = tree_unflatten(example_inputs_flattened, input_shape_spec)
        yield (sample_to_condition(all_mutables.simplify(), sample), sampled_module, sampled_input)

class NnMeterProfiler(ExpressionProfiler):
    """
    Profiler based on `nnMeter <https://github.com/microsoft/nn-Meter>`__,
    which is a tool to estimate the latency of neural networks without real device.

    The profiler breaks the whole model into submodules and profiles each of them,
    introducing branches when some part of the model contains mutables.
    The latency of a module is the sum of the latency of its submodules.

    :class:`NnMeterProfiler` does not respect :func:`~nni.nas.profiler.pytorch.utils.is_leaf_module`
    when it profiles the latency of the model space.
    To control the granularity, inherit this class and override :meth:`is_leaf_module`.

    Parameters
    ----------
    model_space
        The model space to profile.
    args
        Dummy inputs to the model to count flops.
        Similar to `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`__,
        the input can be a tensor or a tuple of tensors, or a tuple of arguments ends with a dictionary of keyword arguments.
    predictor
        The latency predictor to use. Can be a string (alias of nnMeterPredictor) or a :class:`nnMeterPredictor`.
    custom_leaf_types
        A tuple of types of modules that should be considered as leaf modules.
    simplify_shapes
        Experimental feature. If True, the shapes of the inputs and outputs of each module
        will be mathematically simplified with the underlying sympy library.
    """

    def __init__(self, model_space: ModelSpace, args: Any, predictor: str | nnMeterPredictor, custom_leaf_types: tuple[type, ...] | None=None, simplify_shapes: bool=False):
        if False:
            i = 10
            return i + 15
        (args, kwargs) = standardize_arguments(args, lambda t: ShapeTensor(t, True))
        shapes = submodule_input_output_shapes(model_space, *args, **kwargs)
        if simplify_shapes:
            from .utils._expression import recursive_simplification
            shapes = recursive_simplification(shapes)
        self.predictor = load_latency_predictor(predictor) if isinstance(predictor, str) else predictor
        self.custom_leaf_types = custom_leaf_types
        self.expression = self.estimate_latency('', model_space, shapes)

    def is_leaf_module(self, module: nn.Module) -> bool:
        if False:
            return 10
        'If this method returns true for a module,\n        the profiler will exhaust all the possible freeze result of the module,\n        and gets each latency respectively.\n\n        By default, it returns true for modules where :func:`~nni.nas.profiler.pytorch.utils.is_leaf_module` returns true,\n        or for :class:`~nni.nas.nn.pytorch.MutableModule` but not a :class:`~nni.nas.nn.pytorch.LayerChoice` or\n        :class:`~nni.nas.nn.pytorch.Repeat` or a model space without dangling mutables.\n        '
        if self.custom_leaf_types is not None and isinstance(module, self.custom_leaf_types):
            return True
        return is_leaf_module(module) or (isinstance(module, MutableModule) and (not isinstance(module, LayerChoice)) and (not isinstance(module, Repeat)) and (not (isinstance(module, ModelSpace) and (not module.mutables))))

    def exhaust_combinations(self, name: str, module: nn.Module, shapes: dict[str, Any]) -> MutableExpression[float] | float:
        if False:
            print('Hello World!')
        all_latency_cases = []
        if name not in shapes:
            raise ValueError(f'Cannot find input shape for module "{name}"')
        (input_shape, _) = shapes[name]
        for (counter, (condition, sample_module, inputs)) in enumerate(combinations(module, input_shape)):
            if counter >= 100 and counter % 100 == 0:
                _logger.warning('Leaf module named %s contains over %d possibilities.\nCurrent condition: %s\nCurrent sampled module: %s', name, counter, condition, sample_module)
            onnx_model = to_onnx(sample_module, inputs)
            try:
                latency = self.predictor.predict(onnx_model, model_type='onnx')
            except:
                _logger.error('Failed to predict latency for module "%s":\n%s\nThis is likely to be an unsupported case of nn-Meter. To identify the reason, please try to use nn-Meter to predict its latency directly. You can either extend nn-Meter by yourself, or contact nn-Meter team for support, or rewrite your module to work around.', name, sample_module)
                raise
            all_latency_cases.append((condition, latency))
        return MutableExpression.case(all_latency_cases)

    def estimate_layerchoice_latency(self, name: str, module: LayerChoice, shapes: dict[str, Any]) -> MutableExpression[float]:
        if False:
            i = 10
            return i + 15
        'Estimate the latency of a layer choice.\n\n        Profile each choice block and merge them into a switch-case expression.\n        '
        sub_results: dict[int | str, MutableExpression[float] | float] = {}
        for sample_val in module.choice.values:
            latency = self.estimate_latency(concat_name(name, str(sample_val)), module[sample_val], shapes)
            sub_results[sample_val] = latency
        return MutableExpression.switch_case(module.choice, sub_results)

    def estimate_repeat_latency(self, name: str, module: Repeat, shapes: dict[str, Any]) -> MutableExpression[float] | float:
        if False:
            i = 10
            return i + 15
        'Estimate the latency of a Repeat.\n\n        Profile each block and merge possibilities at different depths into a switch-case expression.\n        '
        if isinstance(module.depth_choice, int):
            return sum((self.estimate_latency(concat_name(name, n), child, shapes) for (n, child) in module.named_children()))
        else:
            sub_results: list[MutableExpression] = []
            for (depth, sub) in enumerate(module.blocks, start=1):
                sub_results.append((module.depth_choice >= depth) * self.estimate_latency(concat_name(name, f'blocks.{depth - 1}'), sub, shapes))
            return sum(sub_results)

    def estimate_latency(self, name: str, module: nn.Module, shapes: dict[str, Any]) -> MutableExpression[float] | float:
        if False:
            while True:
                i = 10
        'Count the latency of a mutable module with the given mutable input shapes.\n\n        Returns a mutable expression that is the template of the latency.\n\n        Parameters\n        ----------\n        name\n            The name of the module.\n        module\n            The module to count latency.\n        shapes\n            The input shapes to the module.\n        '
        if self.is_leaf_module(module):
            _logger.debug('NnMeterProfiler finds leaf module: "%s" (type %s). Exhausting all the combinations...', name, type(module).__name__)
            return self.exhaust_combinations(name, module, shapes)
        elif isinstance(module, LayerChoice):
            return self.estimate_layerchoice_latency(name, module, shapes)
        elif isinstance(module, Repeat):
            return self.estimate_repeat_latency(name, module, shapes)
        else:
            return sum((self.estimate_latency(concat_name(name, n), child, shapes) for (n, child) in module.named_children()))