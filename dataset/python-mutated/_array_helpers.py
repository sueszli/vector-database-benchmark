import re
from typing import NamedTuple, Optional, Tuple, Union
from hypothesis import assume, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier, not_set
__all__ = ['NDIM_MAX', 'Shape', 'BroadcastableShapes', 'BasicIndex', 'check_argument', 'order_check', 'check_valid_dims', 'array_shapes', 'valid_tuple_axes', 'broadcastable_shapes', 'mutually_broadcastable_shapes', 'MutuallyBroadcastableShapesStrategy', 'BasicIndexStrategy']
Shape = Tuple[int, ...]
BasicIndex = Tuple[Union[int, slice, None, 'ellipsis'], ...]

class BroadcastableShapes(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape

@check_function
def check_argument(condition, fail_message, *f_args, **f_kwargs):
    if False:
        i = 10
        return i + 15
    if not condition:
        raise InvalidArgument(fail_message.format(*f_args, **f_kwargs))

@check_function
def order_check(name, floor, min_, max_):
    if False:
        print('Hello World!')
    if floor > min_:
        raise InvalidArgument(f'min_{name} must be at least {floor} but was {min_}')
    if min_ > max_:
        raise InvalidArgument(f'min_{name}={min_} is larger than max_{name}={max_}')
NDIM_MAX = 32

@check_function
def check_valid_dims(dims, name):
    if False:
        print('Hello World!')
    if dims > NDIM_MAX:
        raise InvalidArgument(f'{name}={dims}, but Hypothesis does not support arrays with more than {NDIM_MAX} dimensions')

@defines_strategy()
def array_shapes(*, min_dims: int=1, max_dims: Optional[int]=None, min_side: int=1, max_side: Optional[int]=None) -> st.SearchStrategy[Shape]:
    if False:
        i = 10
        return i + 15
    'Return a strategy for array shapes (tuples of int >= 1).\n\n    * ``min_dims`` is the smallest length that the generated shape can possess.\n    * ``max_dims`` is the largest length that the generated shape can possess,\n      defaulting to ``min_dims + 2``.\n    * ``min_side`` is the smallest size that a dimension can possess.\n    * ``max_side`` is the largest size that a dimension can possess,\n      defaulting to ``min_side + 5``.\n    '
    check_type(int, min_dims, 'min_dims')
    check_type(int, min_side, 'min_side')
    check_valid_dims(min_dims, 'min_dims')
    if max_dims is None:
        max_dims = min(min_dims + 2, NDIM_MAX)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = min_side + 5
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    return st.lists(st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims).map(tuple)

@defines_strategy()
def valid_tuple_axes(ndim: int, *, min_size: int=0, max_size: Optional[int]=None) -> st.SearchStrategy[Tuple[int, ...]]:
    if False:
        for i in range(10):
            print('nop')
    'All tuples will have a length >= ``min_size`` and <= ``max_size``. The default\n    value for ``max_size`` is ``ndim``.\n\n    Examples from this strategy shrink towards an empty tuple, which render most\n    sequential functions as no-ops.\n\n    The following are some examples drawn from this strategy.\n\n    .. code-block:: pycon\n\n      >>> [valid_tuple_axes(3).example() for i in range(4)]\n      [(-3, 1), (0, 1, -1), (0, 2), (0, -2, 2)]\n\n    ``valid_tuple_axes`` can be joined with other strategies to generate\n    any type of valid axis object, i.e. integers, tuples, and ``None``:\n\n    .. code-block:: python\n\n      any_axis_strategy = none() | integers(-ndim, ndim - 1) | valid_tuple_axes(ndim)\n\n    '
    check_type(int, ndim, 'ndim')
    check_type(int, min_size, 'min_size')
    if max_size is None:
        max_size = ndim
    check_type(int, max_size, 'max_size')
    order_check('size', 0, min_size, max_size)
    check_valid_interval(max_size, ndim, 'max_size', 'ndim')
    axes = st.integers(0, max(0, 2 * ndim - 1)).map(lambda x: x if x < ndim else x - 2 * ndim)
    return st.lists(axes, min_size=min_size, max_size=max_size, unique_by=lambda x: x % ndim).map(tuple)

@defines_strategy()
def broadcastable_shapes(shape: Shape, *, min_dims: int=0, max_dims: Optional[int]=None, min_side: int=1, max_side: Optional[int]=None) -> st.SearchStrategy[Shape]:
    if False:
        return 10
    'Return a strategy for shapes that are broadcast-compatible with the\n    provided shape.\n\n    Examples from this strategy shrink towards a shape with length ``min_dims``.\n    The size of an aligned dimension shrinks towards size ``1``. The size of an\n    unaligned dimension shrink towards ``min_side``.\n\n    * ``shape`` is a tuple of integers.\n    * ``min_dims`` is the smallest length that the generated shape can possess.\n    * ``max_dims`` is the largest length that the generated shape can possess,\n      defaulting to ``max(len(shape), min_dims) + 2``.\n    * ``min_side`` is the smallest size that an unaligned dimension can possess.\n    * ``max_side`` is the largest size that an unaligned dimension can possess,\n      defaulting to 2 plus the size of the largest aligned dimension.\n\n    The following are some examples drawn from this strategy.\n\n    .. code-block:: pycon\n\n        >>> [broadcastable_shapes(shape=(2, 3)).example() for i in range(5)]\n        [(1, 3), (), (2, 3), (2, 1), (4, 1, 3), (3, )]\n\n    '
    check_type(tuple, shape, 'shape')
    check_type(int, min_side, 'min_side')
    check_type(int, min_dims, 'min_dims')
    check_valid_dims(min_dims, 'min_dims')
    strict_check = max_side is None or max_dims is None
    if max_dims is None:
        max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = max(shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    if strict_check:
        dims = max_dims
        bound_name = 'max_dims'
    else:
        dims = min_dims
        bound_name = 'min_dims'
    if not all((min_side <= s for s in shape[::-1][:dims] if s != 1)):
        raise InvalidArgument(f'Given shape={shape}, there are no broadcast-compatible shapes that satisfy: {bound_name}={dims} and min_side={min_side}')
    if not (min_side <= 1 <= max_side or all((s <= max_side for s in shape[::-1][:dims]))):
        raise InvalidArgument(f'Given base_shape={shape}, there are no broadcast-compatible shapes that satisfy all of {bound_name}={dims}, min_side={min_side}, and max_side={max_side}')
    if not strict_check:
        for (n, s) in zip(range(max_dims), shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break
    return MutuallyBroadcastableShapesStrategy(num_shapes=1, base_shape=shape, min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side).map(lambda x: x.input_shapes[0])
_DIMENSION = '\\w+\\??'
_SHAPE = f'\\((?:{_DIMENSION}(?:,{_DIMENSION}){{0,31}})?\\)'
_ARGUMENT_LIST = f'{_SHAPE}(?:,{_SHAPE})*'
_SIGNATURE = f'^{_ARGUMENT_LIST}->{_SHAPE}$'
_SIGNATURE_MULTIPLE_OUTPUT = f'^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$'

class _GUfuncSig(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape

def _hypothesis_parse_gufunc_signature(signature):
    if False:
        print('Hello World!')
    if not re.match(_SIGNATURE, signature):
        if re.match(_SIGNATURE_MULTIPLE_OUTPUT, signature):
            raise InvalidArgument(f"Hypothesis does not yet support generalised ufunc signatures with multiple output arrays - mostly because we don't know of anyone who uses them!  Please get in touch with us to fix that.\n (signature={signature!r})")
        if re.match('^\\((?:\\w+(?:,\\w+)*)?\\)(?:,\\((?:\\w+(?:,\\w+)*)?\\))*->\\((?:\\w+(?:,\\w+)*)?\\)(?:,\\((?:\\w+(?:,\\w+)*)?\\))*$', signature):
            raise InvalidArgument(f"signature={signature!r} matches Numpy's regex for gufunc signatures, but contains shapes with more than {NDIM_MAX} dimensions and is thus invalid.")
        raise InvalidArgument(f'{signature!r} is not a valid gufunc signature')
    (input_shapes, output_shapes) = (tuple((tuple(re.findall(_DIMENSION, a)) for a in re.findall(_SHAPE, arg_list))) for arg_list in signature.split('->'))
    assert len(output_shapes) == 1
    result_shape = output_shapes[0]
    for shape in (*input_shapes, result_shape):
        for name in shape:
            try:
                int(name.strip('?'))
                if '?' in name:
                    raise InvalidArgument(f'Got dimension {name!r}, but handling of frozen optional dimensions is ambiguous.  If you known how this should work, please contact us to get this fixed and documented ({{signature=}}).')
            except ValueError:
                names_in = {n.strip('?') for shp in input_shapes for n in shp}
                names_out = {n.strip('?') for n in result_shape}
                if name.strip('?') in names_out - names_in:
                    raise InvalidArgument('The {name!r} dimension only appears in the output shape, and is not frozen, so the size is not determined ({signature=}).') from None
    return _GUfuncSig(input_shapes=input_shapes, result_shape=result_shape)

@defines_strategy()
def mutually_broadcastable_shapes(*, num_shapes: Union[UniqueIdentifier, int]=not_set, signature: Union[UniqueIdentifier, str]=not_set, base_shape: Shape=(), min_dims: int=0, max_dims: Optional[int]=None, min_side: int=1, max_side: Optional[int]=None) -> st.SearchStrategy[BroadcastableShapes]:
    if False:
        return 10
    'Return a strategy for a specified number of shapes N that are\n    mutually-broadcastable with one another and with the provided base shape.\n\n    * ``num_shapes`` is the number of mutually broadcast-compatible shapes to generate.\n    * ``base_shape`` is the shape against which all generated shapes can broadcast.\n      The default shape is empty, which corresponds to a scalar and thus does\n      not constrain broadcasting at all.\n    * ``min_dims`` is the smallest length that the generated shape can possess.\n    * ``max_dims`` is the largest length that the generated shape can possess,\n      defaulting to ``max(len(shape), min_dims) + 2``.\n    * ``min_side`` is the smallest size that an unaligned dimension can possess.\n    * ``max_side`` is the largest size that an unaligned dimension can possess,\n      defaulting to 2 plus the size of the largest aligned dimension.\n\n    The strategy will generate a :obj:`python:typing.NamedTuple` containing:\n\n    * ``input_shapes`` as a tuple of the N generated shapes.\n    * ``result_shape`` as the resulting shape produced by broadcasting the N shapes\n      with the base shape.\n\n    The following are some examples drawn from this strategy.\n\n    .. code-block:: pycon\n\n        >>> # Draw three shapes where each shape is broadcast-compatible with (2, 3)\n        ... strat = mutually_broadcastable_shapes(num_shapes=3, base_shape=(2, 3))\n        >>> for _ in range(5):\n        ...     print(strat.example())\n        BroadcastableShapes(input_shapes=((4, 1, 3), (4, 2, 3), ()), result_shape=(4, 2, 3))\n        BroadcastableShapes(input_shapes=((3,), (1, 3), (2, 3)), result_shape=(2, 3))\n        BroadcastableShapes(input_shapes=((), (), ()), result_shape=())\n        BroadcastableShapes(input_shapes=((3,), (), (3,)), result_shape=(3,))\n        BroadcastableShapes(input_shapes=((1, 2, 3), (3,), ()), result_shape=(1, 2, 3))\n\n    '
    arg_msg = 'Pass either the `num_shapes` or the `signature` argument, but not both.'
    if num_shapes is not not_set:
        check_argument(signature is not_set, arg_msg)
        check_type(int, num_shapes, 'num_shapes')
        assert isinstance(num_shapes, int)
        parsed_signature = None
        sig_dims = 0
    else:
        check_argument(signature is not not_set, arg_msg)
        if signature is None:
            raise InvalidArgument('Expected a string, but got invalid signature=None.  (maybe .signature attribute of an element-wise ufunc?)')
        check_type(str, signature, 'signature')
        parsed_signature = _hypothesis_parse_gufunc_signature(signature)
        all_shapes = (*parsed_signature.input_shapes, parsed_signature.result_shape)
        sig_dims = min((len(s) for s in all_shapes))
        num_shapes = len(parsed_signature.input_shapes)
    if num_shapes < 1:
        raise InvalidArgument(f'num_shapes={num_shapes} must be at least 1')
    check_type(tuple, base_shape, 'base_shape')
    check_type(int, min_side, 'min_side')
    check_type(int, min_dims, 'min_dims')
    check_valid_dims(min_dims, 'min_dims')
    strict_check = max_dims is not None
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 2, NDIM_MAX - sig_dims)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    if signature is not None and max_dims > NDIM_MAX - sig_dims:
        raise InvalidArgument(f'max_dims={signature!r} would exceed the {NDIM_MAX}-dimensionlimit Hypothesis imposes on array shapes, given signature={parsed_signature!r}')
    if strict_check:
        dims = max_dims
        bound_name = 'max_dims'
    else:
        dims = min_dims
        bound_name = 'min_dims'
    if not all((min_side <= s for s in base_shape[::-1][:dims] if s != 1)):
        raise InvalidArgument(f'Given base_shape={base_shape}, there are no broadcast-compatible shapes that satisfy: {bound_name}={dims} and min_side={min_side}')
    if not (min_side <= 1 <= max_side or all((s <= max_side for s in base_shape[::-1][:dims]))):
        raise InvalidArgument(f'Given base_shape={base_shape}, there are no broadcast-compatible shapes that satisfy all of {bound_name}={dims}, min_side={min_side}, and max_side={max_side}')
    if not strict_check:
        for (n, s) in zip(range(max_dims), base_shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break
    return MutuallyBroadcastableShapesStrategy(num_shapes=num_shapes, signature=parsed_signature, base_shape=base_shape, min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side)

class MutuallyBroadcastableShapesStrategy(st.SearchStrategy):

    def __init__(self, num_shapes, signature=None, base_shape=(), min_dims=0, max_dims=None, min_side=1, max_side=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.base_shape = base_shape
        self.side_strat = st.integers(min_side, max_side)
        self.num_shapes = num_shapes
        self.signature = signature
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.min_side = min_side
        self.max_side = max_side
        self.size_one_allowed = self.min_side <= 1 <= self.max_side

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        if self.signature is None:
            return self._draw_loop_dimensions(data)
        (core_in, core_res) = self._draw_core_dimensions(data)
        use = [None not in shp for shp in core_in]
        (loop_in, loop_res) = self._draw_loop_dimensions(data, use=use)

        def add_shape(loop, core):
            if False:
                for i in range(10):
                    print('nop')
            return tuple((x for x in (loop + core)[-NDIM_MAX:] if x is not None))
        return BroadcastableShapes(input_shapes=tuple((add_shape(l_in, c) for (l_in, c) in zip(loop_in, core_in))), result_shape=add_shape(loop_res, core_res))

    def _draw_core_dimensions(self, data):
        if False:
            while True:
                i = 10
        dims = {}
        shapes = []
        for shape in (*self.signature.input_shapes, self.signature.result_shape):
            shapes.append([])
            for name in shape:
                if name.isdigit():
                    shapes[-1].append(int(name))
                    continue
                if name not in dims:
                    dim = name.strip('?')
                    dims[dim] = data.draw(self.side_strat)
                    if self.min_dims == 0 and (not data.draw_bits(3)):
                        dims[dim + '?'] = None
                    else:
                        dims[dim + '?'] = dims[dim]
                shapes[-1].append(dims[name])
        return (tuple((tuple(s) for s in shapes[:-1])), tuple(shapes[-1]))

    def _draw_loop_dimensions(self, data, use=None):
        if False:
            return 10
        base_shape = self.base_shape[::-1]
        result_shape = list(base_shape)
        shapes = [[] for _ in range(self.num_shapes)]
        if use is None:
            use = [True for _ in range(self.num_shapes)]
        else:
            assert len(use) == self.num_shapes
            assert all((isinstance(x, bool) for x in use))
        for dim_count in range(1, self.max_dims + 1):
            dim = dim_count - 1
            if len(base_shape) < dim_count or base_shape[dim] == 1:
                dim_side = data.draw(self.side_strat)
            elif base_shape[dim] <= self.max_side:
                dim_side = base_shape[dim]
            else:
                dim_side = 1
            allowed_sides = sorted([1, dim_side])
            for (shape_id, shape) in enumerate(shapes):
                if dim <= len(result_shape) and self.size_one_allowed:
                    side = data.draw(st.sampled_from(allowed_sides))
                else:
                    side = dim_side
                if self.min_dims < dim_count:
                    use[shape_id] &= cu.biased_coin(data, 1 - 1 / (1 + self.max_dims - dim))
                if use[shape_id]:
                    shape.append(side)
                    if len(result_shape) < len(shape):
                        result_shape.append(shape[-1])
                    elif shape[-1] != 1 and result_shape[dim] == 1:
                        result_shape[dim] = shape[-1]
            if not any(use):
                break
        result_shape = result_shape[:max(map(len, [self.base_shape, *shapes]))]
        assert len(shapes) == self.num_shapes
        assert all((self.min_dims <= len(s) <= self.max_dims for s in shapes))
        assert all((self.min_side <= s <= self.max_side for side in shapes for s in side))
        return BroadcastableShapes(input_shapes=tuple((tuple(reversed(shape)) for shape in shapes)), result_shape=tuple(reversed(result_shape)))

class BasicIndexStrategy(st.SearchStrategy):

    def __init__(self, shape, min_dims, max_dims, allow_ellipsis, allow_newaxis, allow_fewer_indices_than_dims):
        if False:
            print('Hello World!')
        self.shape = shape
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.allow_ellipsis = allow_ellipsis
        self.allow_newaxis = allow_newaxis
        self.allow_fewer_indices_than_dims = allow_fewer_indices_than_dims

    def do_draw(self, data):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for dim_size in self.shape:
            if dim_size == 0:
                result.append(slice(None))
                continue
            strategy = st.integers(-dim_size, dim_size - 1) | st.slices(dim_size)
            result.append(data.draw(strategy))
        result_dims = sum((isinstance(idx, slice) for idx in result))
        while self.allow_newaxis and result_dims < self.max_dims and (result_dims < self.min_dims or data.draw(st.booleans())):
            i = data.draw(st.integers(0, len(result)))
            result.insert(i, None)
            result_dims += 1
        assume(self.min_dims <= result_dims <= self.max_dims)
        if self.allow_ellipsis and data.draw(st.booleans()):
            i = j = data.draw(st.integers(0, len(result)))
            while i > 0 and result[i - 1] == slice(None):
                i -= 1
            while j < len(result) and result[j] == slice(None):
                j += 1
            result[i:j] = [Ellipsis]
        elif self.allow_fewer_indices_than_dims:
            while result[-1:] == [slice(None, None)] and data.draw(st.integers(0, 7)):
                result.pop()
        if len(result) == 1 and data.draw(st.booleans()):
            return result[0]
        return tuple(result)