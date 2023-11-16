"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""
from __future__ import annotations
from typing import Any, cast
import torch
import torch.nn as nn
from nni.mutable import Categorical, Mutable, Sample, SampleValidationError, MutableExpression, label_scope, auto_label
from nni.nas.nn.pytorch import Repeat, recursive_freeze
from .base import BaseSuperNetModule
from .differentiable import DifferentiableMixedLayer, DifferentiableMixedInput
from ._expression_utils import traverse_all_options
__all__ = ['ProxylessMixedLayer', 'ProxylessMixedInput', 'ProxylessMixedRepeat', 'suppress_already_mutated']

def _detach_tensor(tensor: Any, requires_grad: bool=False) -> Any:
    if False:
        i = 10
        return i + 15
    'Recursively detach all the tensors.'
    if isinstance(tensor, (list, tuple)):
        return tuple((_detach_tensor(t, requires_grad) for t in tensor))
    elif isinstance(tensor, dict):
        return {k: _detach_tensor(v, requires_grad) for (k, v) in tensor.items()}
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if requires_grad:
            tensor.requires_grad_()
        return tensor
    else:
        return tensor

def _iter_tensors(tensor: Any) -> Any:
    if False:
        i = 10
        return i + 15
    'Recursively iterate over all the tensors.\n\n    This is kept for complex outputs (like dicts / lists).\n    However, complex outputs are not supported by PyTorch backward hooks yet.\n    '
    if isinstance(tensor, torch.Tensor):
        yield tensor
    elif isinstance(tensor, (list, tuple)):
        for t in tensor:
            yield from _iter_tensors(t)
    elif isinstance(tensor, dict):
        for t in tensor.values():
            yield from _iter_tensors(t)

def _pack_as_tuple(tensor: Any) -> tuple:
    if False:
        print('Hello World!')
    "Return a tuple of tensor with only one element if tensor it's not a tuple."
    if isinstance(tensor, (tuple, list)):
        for t in tensor:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f'All elements in the tuple must be of the same type (tensor), but got {type(t)}')
        return tuple(tensor)
    elif isinstance(tensor, torch.Tensor):
        return (tensor,)
    else:
        raise TypeError(f'Unsupported type {type(tensor)}')

def _unpack_tuple(tensor: tuple) -> Any:
    if False:
        print('Hello World!')
    'Return a single element if a single-element tuple. Otherwise a tuple.'
    if len(tensor) == 1:
        return tensor[0]
    else:
        return tensor

def element_product_sum(tensor1: tuple[torch.Tensor, ...], tensor2: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if False:
        print('Hello World!')
    'Compute the sum of all the element-wise product.'
    assert len(tensor1) == len(tensor2), 'The number of tensors must be the same.'
    ret = [torch.sum(t1 * t2) for (t1, t2) in zip(tensor1, tensor2) if t1 is not None and t2 is not None]
    if not ret:
        return torch.tensor(0)
    if len(ret) == 1:
        return ret[0]
    return cast(torch.Tensor, sum(ret))

class _ProxylessFunction(torch.autograd.Function):
    """
    Compute the gradient of ``arch_alpha``,
    given the sampled index, inputs and outputs of the layer, and a function to
    compute the output of the layer conditioned on any other index.

    The forward of this function is merely returning a "copy" of the given input.
    This is different from the
    `official implementation <https://github.com/mit-han-lab/proxylessnas/blob/9cdd0791/search/modules/mix_op.py>`,
    where the output is computed within this function.

    Things tried but NOT WORKING:

    1. Use ``full_backward_hook`` instead of ``autograd.Function``.
       Since ``full_backward_hook`` is not intended for computing gradients for parameters,
       The gradients get overridden when another loss is added (e.g., latency loss).
    2. Computing the output in ``autograd.Function`` like the official impl does.
       This requires calling either ``autograd.grad`` (doesn't work when nested),
       or ``autograd.backward`` in the backward of the Function class.
       The gradients within the candidates (like Linear, or Conv2d) will computed inside this function.
       This is problematic with DDP, because either (i) with ``find_unused_parameters=True``,
       the parameters within the candidates are considered unused (because they are not found on the autograd graph),
       and raises error when they receive a gradient (something like "another gradient received after ready"), or
       (ii) with ``find_unused_parameters=False``, DDP complains about some candidate paths having no gradients at all.

    Therefore, the only way to make it work is to write a function::

        func(alpha, input, output) = output

    From the outside of this box, ``func`` echoes the output. But inside, there is gradient going on.
    When back-propagation, ``alpha`` will receive a computed gradient.
    ``input`` receives none because it didn't participate in the computation
    (but only needs to be saved for computing gradients).
    ``output`` receives the gradient that ``func`` receives.
    """

    @staticmethod
    def forward(ctx, forward_path_func, sample_index, num_inputs, softmax, arch_alpha, *layer_input_output):
        if False:
            while True:
                i = 10
        ctx.forward_path_func = forward_path_func
        ctx.sample_index = sample_index
        ctx.num_inputs = num_inputs
        ctx.softmax = softmax
        ctx.save_for_backward(arch_alpha, *layer_input_output)
        layer_output = layer_input_output[num_inputs:]
        return _unpack_tuple(_detach_tensor(layer_output, requires_grad=True))

    @staticmethod
    def backward(ctx, *grad_output):
        if False:
            for i in range(10):
                print('nop')
        (softmax, sample_index, forward_path_func, num_inputs) = (ctx.softmax, ctx.sample_index, ctx.forward_path_func, ctx.num_inputs)
        if ctx.needs_input_grad[0]:
            grads = None
        else:
            (arch_alpha, *layer_input_output) = ctx.saved_tensors
            layer_input = layer_input_output[:num_inputs]
            layer_output = layer_input_output[num_inputs:]
            binary_grads = torch.zeros_like(arch_alpha)
            with torch.no_grad():
                for k in range(len(binary_grads)):
                    if k != sample_index:
                        out_k = forward_path_func(k, *layer_input)
                    else:
                        out_k = layer_output
                    binary_grads[k] = element_product_sum(_pack_as_tuple(out_k), grad_output)
                grads = torch.zeros_like(arch_alpha)
                probs = softmax(arch_alpha)
                for i in range(len(arch_alpha)):
                    for j in range(len(arch_alpha)):
                        grads[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])
        return (None, None, None, None, grads, *[None] * num_inputs, *grad_output)

def suppress_already_mutated(module, name, memo, mutate_kwargs) -> bool | None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(module, (ProxylessMixedLayer, ProxylessMixedInput, ProxylessMixedRepeat)):
        return True
    return None

class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than compute the weighted sum.

    Currently the input and output of the candidate layers can only be tensors or tuple of tensors.
    They can't be dict, list or any complex types, or non-tensors (including none).
    """
    _arch_parameter_names = ['_arch_alpha']

    def __init__(self, paths: dict[str, nn.Module], alpha: torch.Tensor, softmax: nn.Module, label: str):
        if False:
            print('Hello World!')
        super().__init__(paths, alpha, softmax, label)
        self._sampled: str | int | None = None
        self._sample_idx: int | None = None

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Forward pass of one single path.'
        if self._sample_idx is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        if kwargs:
            raise ValueError(f'kwargs is not supported yet in {self.__class__.__name__}.')
        result = self.forward_path(self._sample_idx, *args, **kwargs)
        return _ProxylessFunction.apply(self.forward_path, self._sample_idx, len(args), self._softmax, self._arch_alpha, *args, *_pack_as_tuple(result))

    def forward_path(self, index, *args, **kwargs):
        if False:
            return 10
        return self[self.names[index]](*args, **kwargs)

    def resample(self, memo):
        if False:
            print('Hello World!')
        'Sample one path based on alpha if label is not found in memo.'
        if self.label in memo:
            self._sampled = memo[self.label]
            self._sample_idx = self.names.index(self._sampled)
        else:
            probs = self._softmax(self._arch_alpha)
            self._sample_idx = int(torch.multinomial(probs, 1)[0].item())
            self._sampled = self.names[self._sample_idx]
        return {self.label: self._sampled}

    def export(self, memo):
        if False:
            print('Hello World!')
        'Same as :meth:`resample`.'
        return self.resample(memo)

class ProxylessMixedInput(DifferentiableMixedInput):
    """Proxyless version of differentiable input choice.
    See :class:`ProxylessMixedLayer` for implementation details.
    """
    _arch_parameter_names = ['_arch_alpha', '_binary_gates']

    def __init__(self, n_candidates: int, n_chosen: int | None, alpha: torch.Tensor, softmax: nn.Module, label: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(n_candidates, n_chosen, alpha, softmax, label)
        self._sampled: list[int] | None = None

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        'Choose one single input.'
        if self._sampled is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        result = self.forward_path(self._sampled[0], *inputs)
        return _ProxylessFunction.apply(self.forward_path, self._sampled[0], len(inputs), self._softmax, self._arch_alpha, *inputs, result)

    def forward_path(self, index, *inputs):
        if False:
            for i in range(10):
                print('nop')
        return inputs[index]

    def resample(self, memo):
        if False:
            return 10
        'Sample one path based on alpha if label is not found in memo.'
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self._softmax(self._arch_alpha)
            n_chosen = self.n_chosen or 1
            sample = torch.multinomial(probs, n_chosen).cpu().numpy().tolist()
            self._sampled = sample
        return {self.label: self._sampled}

    def export(self, memo):
        if False:
            print('Hello World!')
        'Same as :meth:`resample`.'
        return self.resample(memo)

class ProxylessMixedRepeat(Repeat, BaseSuperNetModule):
    """ProxylessNAS converts repeat to a sequential blocks of layer choices between
    the original block and an identity layer.

    Only pure categorical depth choice is supported.
    If the categorical choices are not consecutive integers, the constraint will only be considered at export.
    """
    depth_choice: Categorical[int]

    def __init__(self, blocks: list[nn.Module], depth: Categorical[int]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(blocks, depth)
        assert isinstance(depth, Categorical)
        assert len(blocks) == self.max_depth
        for d in range(self.min_depth, self.max_depth):
            block = blocks[d]
            assert isinstance(block, ProxylessMixedLayer)
            assert len(block._arch_alpha) == 2

    def resample(self, memo):
        if False:
            print('Hello World!')
        'Resample each individual depths.'
        if self.depth_choice.label in memo:
            return {}
        depth = self.min_depth
        for d in range(self.min_depth, self.max_depth):
            layer = self.blocks[d]
            assert isinstance(layer, ProxylessMixedLayer)
            memo.pop(layer.label, None)
            sample = layer.resample(memo)
            memo.update(sample)
            depth += int(memo[layer.label])
        return {self.depth_choice.label: depth}

    def export(self, memo):
        if False:
            while True:
                i = 10
        'Return the most likely to be chosen depth choice.'
        sample = {}
        for _ in range(1000):
            sample = self.resample(memo)
            if sample[self.depth_choice.label] in self.depth_choice.values:
                return sample
        return sample

    def export_probs(self, memo):
        if False:
            return 10
        'Compute the probability of each depth choice gets chosen.'
        if self.depth_choice.label in memo:
            return {}
        categoricals: list[Categorical] = []
        weights: dict[str, torch.Tensor] = {}
        for d in range(self.min_depth, self.max_depth):
            layer = cast(ProxylessMixedLayer, self.blocks[d])
            categoricals.append(MutableExpression.to_int(layer.choice))
            weights[layer.label] = layer._softmax(layer._arch_alpha)
        return {self.depth_choice.label: dict(traverse_all_options(cast(MutableExpression[int], sum(categoricals) + self.min_depth), weights))}

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        if False:
            for i in range(10):
                print('nop')
        exception = self.depth_choice.check_contains(sample)
        if exception is not None:
            return exception
        depth = self.depth_choice.freeze(sample)
        for (i, block) in enumerate(self.blocks):
            if i < self.min_depth:
                exception = self._check_any_module_contains(block, sample, str(i))
            elif i < depth:
                assert isinstance(block, ProxylessMixedLayer)
                exception = self._check_any_module_contains(block['1'], sample, str(i))
            else:
                break
        return None

    def freeze(self, sample: Sample) -> nn.Sequential:
        if False:
            i = 10
            return i + 15
        self.validate(sample)
        depth = self.depth_choice.freeze(sample)
        blocks = []
        for (i, block) in enumerate(self.blocks):
            if i < self.min_depth:
                blocks.append(recursive_freeze(block, sample)[0])
            elif i < depth:
                assert isinstance(block, ProxylessMixedLayer)
                blocks.append(recursive_freeze(block['1'], sample)[0])
            else:
                break
        return nn.Sequential(*blocks)

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if False:
            i = 10
            return i + 15
        if type(module) == Repeat and isinstance(module.depth_choice, Mutable):
            module = cast(Repeat, module)
            if not isinstance(module.depth_choice, Categorical):
                raise ValueError(f'The depth choice must be a straightforward categorical, but got {module.depth_choice}')
            blocks: list[nn.Module] = []
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            with label_scope(module.depth_choice.label):
                for (i, block) in enumerate(module.blocks):
                    if i < module.min_depth:
                        blocks.append(block)
                    else:
                        label = auto_label(f'in_repeat_{i}')
                        if label in memo:
                            alpha = memo[label]
                        else:
                            alpha = nn.Parameter(torch.randn(2) * 0.001)
                            memo[label] = alpha
                        candidates = {'0': nn.Identity(), '1': block}
                        blocks.append(ProxylessMixedLayer(candidates, alpha, softmax, label))
            return cls(blocks, module.depth_choice)

    def forward(self, x):
        if False:
            print('Hello World!')
        for block in self.blocks:
            x = block(x)
        return x