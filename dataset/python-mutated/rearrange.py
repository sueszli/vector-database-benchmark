from __future__ import annotations
import functools
from typing import Callable, Dict, List, Sequence, Tuple, Union
import torch
from functorch._C import dim as _C
from ._parsing import _ellipsis, AnonymousAxis, comma_separate, parse_pattern, validate_rearrange_expressions
__all__ = ['rearrange']
dims = _C.dims

@functools.lru_cache(256)
def _create_rearrange_callable(tensor_ndim: int, pattern: str, **axes_lengths: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if False:
        return 10
    'Translate an `einops`-style pattern into a callable that performs the rearrange using first-class dimensions.\n\n    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and\n    specified axes lengths, this function can be memoized.\n\n    Args:\n        tensor_ndim (int): the number of dimensions in the tensor to rearrange\n        pattern (str): the `einops`-style rearrangement pattern\n        axes_lengths (int): any additional length specifications for dimensions\n\n    Returns:\n        Callable[[torch.Tensor], torch.Tensor]: a callable that performs the rearrangement\n    '
    (left, right) = parse_pattern(pattern, axes_lengths)
    validate_rearrange_expressions(left, right, axes_lengths)
    n_anon_dims = sum((not dim for dim in left.composition))
    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_named_dims = len(left.identifiers) - 1
        if (pattern_ndim := (n_anon_dims + n_named_dims)) > tensor_ndim:
            raise ValueError(f'Number of dimensions in pattern ({pattern_ndim}) must be less than or equal to the number of dimensions in the tensor ({tensor_ndim})')
    else:
        n_ellipsis_dims = 0
        n_named_dims = len(left.identifiers)
        if (pattern_ndim := len(left.composition)) != tensor_ndim:
            raise ValueError(f'Number of dimensions in pattern ({pattern_ndim}) must be equal to the number of dimensions in the tensor ({tensor_ndim})')
    n_dims = n_named_dims + n_ellipsis_dims + n_anon_dims
    if n_dims == 0:
        return lambda tensor: tensor
    first_class_dims: Tuple[str, ...] = tuple((f'd{i}' for i in range(n_dims)))
    identifier_dim_map: Dict[Union[str, AnonymousAxis], Tuple[str, ...]] = {}
    anon_axes: List[AnonymousAxis] = []
    dims_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                assert isinstance(identifier, str)
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
            if not dimension:
                anon_axis = AnonymousAxis('1')
                identifier_dim_map[anon_axis] = (first_class_dims[dims_i],)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dims_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_dim_map[identifier] = tuple((first_class_dims[dims_i + j] for j in range(n_ellipsis_dims)))
            dims_i += n_ellipsis_dims
        else:
            raise ValueError(f'Unexpected dimension: {dimension}')

    def composition_to_dims(composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]) -> List[Union[str, Tuple[str, ...]]]:
        if False:
            i = 10
            return i + 15
        'Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first\n        class dims.'
        dim_composition: List[Union[str, Tuple[str, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                dim_composition.append(tuple((dim for identifier in dimension for dim in identifier_dim_map[identifier])))
            elif dimension == _ellipsis:
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                raise ValueError(f'Unexpected dimension: {dimension}')
        return dim_composition
    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    anon_dims = tuple((identifier_dim_map[axis][0] for axis in anon_axes))
    specified_lengths = tuple(((identifier_dim_map[axis][0], length) for (axis, length) in axes_lengths.items()))
    custom_rearrange_callable_name = 'do_rearrange'
    custom_rearrange_callable_code = f'def {custom_rearrange_callable_name}(tensor):\n    {comma_separate(first_class_dims)} = dims({n_dims})\n' + (''.join((f'    {dim}.size = {length}\n' for (dim, length) in specified_lengths)) if specified_lengths else '') + f'    tensor = tensor[{comma_separate(left_dims)}].order({comma_separate(right_dims)})\n' + (f'    return tensor.sum({comma_separate([anon_dims])}, keepdim=False)\n' if anon_dims else '    return tensor\n')
    exec(custom_rearrange_callable_code)
    return locals()[custom_rearrange_callable_name]

def rearrange(tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]], pattern: str, **axes_lengths: int) -> torch.Tensor:
    if False:
        print('Hello World!')
    'A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional\n    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,\n    stack, concatenate and other operations.\n\n    See: https://einops.rocks/api/rearrange/\n\n    Args:\n        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange\n        pattern (str): the rearrangement pattern\n        axes_lengths (int): any additional length specifications for dimensions\n\n    Returns:\n        Tensor: the rearranged tensor\n\n    Examples:\n        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)\n        >>> images = torch.randn((32, 30, 40, 3))\n\n        >>> # stack along first (batch) axis, output is a single array\n        >>> rearrange(images, \'b h w c -> b h w c\').shape\n        torch.Size([32, 30, 40, 3])\n\n        >>> # concatenate images along height (vertical axis), 960 = 32 * 30\n        >>> rearrange(images, \'b h w c -> (b h) w c\').shape\n        torch.Size([960, 40, 3])\n\n        >>> # concatenated images along horizontal axis, 1280 = 32 * 40\n        >>> rearrange(images, \'b h w c -> h (b w) c\').shape\n        torch.Size([30, 1280, 3])\n\n        >>> # reordered axes to "b c h w" format for deep learning\n        >>> rearrange(images, \'b h w c -> b c h w\').shape\n        torch.Size([32, 3, 30, 40])\n\n        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3\n        >>> rearrange(images, \'b h w c -> b (c h w)\').shape\n        torch.Size([32, 3600])\n\n        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2\n        >>> rearrange(images, \'b (h1 h) (w1 w) c -> (b h1 w1) h w c\', h1=2, w1=2).shape\n        torch.Size([128, 15, 20, 3])\n\n        >>> # space-to-depth operation\n        >>> rearrange(images, \'b (h h1) (w w1) c -> b h w (c h1 w1)\', h1=2, w1=2).shape\n        torch.Size([32, 15, 20, 12])\n    '
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)
    rearrange_callable = _create_rearrange_callable(tensor.ndim, pattern, **axes_lengths)
    return rearrange_callable(tensor)