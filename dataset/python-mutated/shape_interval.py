from typing import Sequence
from deeplake.util.exceptions import InvalidShapeIntervalError
from typing import Optional, Sequence, Tuple

def _contains_negatives(shape: Sequence[int]):
    if False:
        i = 10
        return i + 15
    return any((x and x < 0 for x in shape))

class ShapeInterval:

    def __init__(self, lower: Sequence[int], upper: Optional[Sequence[int]]=None):
        if False:
            while True:
                i = 10
        '\n        Shapes in Deep Lake are best represented as intervals, this is to support dynamic tensors. Instead of having a single tuple of integers representing shape,\n        we use 2 tuples of integers to represent the lower and upper bounds of the representing shape.\n\n        - If ``lower == upper`` for all cases, the shape is considered "fixed".\n        - If ``lower != upper`` for any cases, the shape is considered "dynamic".\n\n        Args:\n            lower (sequence): Sequence of integers that represent the lower-bound shape.\n            upper (sequence): Sequence of integers that represent the upper-bound shape. If None is provided, lower is used as upper (implicitly fixed-shape).\n\n        Raises:\n            InvalidShapeIntervalError: If the provided lower/upper bounds are incompatible to represent a shape.\n        '
        if upper is None:
            upper = lower
        if len(lower) != len(upper):
            raise InvalidShapeIntervalError('Lengths must match.', lower, upper)
        if _contains_negatives(lower):
            raise InvalidShapeIntervalError('Lower cannot contain negative components.', lower=lower)
        if _contains_negatives(upper):
            raise InvalidShapeIntervalError('Upper cannot contain negative components.', upper=upper)
        if not all((l is None or u is None or l <= u for (l, u) in zip(lower, upper))):
            raise InvalidShapeIntervalError('lower[i] must always be <= upper[i].', lower=lower, upper=upper)
        self._lower = tuple(lower)
        self._upper = tuple(upper)

    def astuple(self) -> Tuple[Optional[int], ...]:
        if False:
            for i in range(10):
                print('nop')
        shape = []
        for (low, up) in zip(self.lower, self.upper):
            shape.append(None if low != up else low)
        return tuple(shape)

    @property
    def is_dynamic(self) -> bool:
        if False:
            print('Hello World!')
        return self.lower != self.upper

    @property
    def lower(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        return self._lower

    @property
    def upper(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        return self._upper

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        intervals = []
        for (l, u) in zip(self.lower, self.upper):
            if l == u:
                intervals.append(str(l))
            else:
                intervals.append(f'{l}:{u}')
        if len(intervals) == 1:
            return f'({intervals[0]},)'
        return f"({', '.join(intervals)})"

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, ShapeInterval):
            return False
        return self.lower == other.lower and self.upper == other.upper