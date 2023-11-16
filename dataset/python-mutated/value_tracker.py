from __future__ import annotations
import numpy as np
from manimlib.mobject.mobject import Mobject
from manimlib.utils.iterables import listify
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import Self

class ValueTracker(Mobject):
    """
    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.
    """
    value_type: type = np.float64

    def __init__(self, value: float | complex | np.ndarray=0, **kwargs):
        if False:
            return 10
        self.value = value
        super().__init__(**kwargs)

    def init_uniforms(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().init_uniforms()
        self.uniforms['value'] = np.array(listify(self.value), dtype=self.value_type)

    def get_value(self) -> float | complex | np.ndarray:
        if False:
            while True:
                i = 10
        result = self.uniforms['value']
        if len(result) == 1:
            return result[0]
        return result

    def set_value(self, value: float | complex | np.ndarray) -> Self:
        if False:
            while True:
                i = 10
        self.uniforms['value'][:] = value
        return self

    def increment_value(self, d_value: float | complex) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.set_value(self.get_value() + d_value)

class ExponentialValueTracker(ValueTracker):
    """
    Operates just like ValueTracker, except it encodes the value as the
    exponential of a position coordinate, which changes how interpolation
    behaves
    """

    def get_value(self) -> float | complex:
        if False:
            return 10
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value: float | complex):
        if False:
            print('Hello World!')
        return ValueTracker.set_value(self, np.log(value))

class ComplexValueTracker(ValueTracker):
    value_type: type = np.complex128