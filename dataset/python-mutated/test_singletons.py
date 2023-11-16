from __future__ import annotations
import pytest
pytest
from copy import copy
import bokeh.core.property.singletons as bcpu
ALL = ('Intrinsic', 'Undefined')

def test_Undefined() -> None:
    if False:
        return 10
    assert (bcpu.Undefined == bcpu.Undefined) is True
    assert (bcpu.Undefined != bcpu.Undefined) is False
    assert (bcpu.Undefined is bcpu.Undefined) is True
    assert (bcpu.Undefined is not bcpu.Undefined) is False
    assert (copy(bcpu.Undefined) is bcpu.Undefined) is True
    assert (copy(bcpu.Undefined) is not bcpu.Undefined) is False

def test_Intrinsic() -> None:
    if False:
        i = 10
        return i + 15
    assert (bcpu.Intrinsic == bcpu.Intrinsic) is True
    assert (bcpu.Intrinsic != bcpu.Intrinsic) is False
    assert (bcpu.Intrinsic is bcpu.Intrinsic) is True
    assert (bcpu.Intrinsic is not bcpu.Intrinsic) is False
    assert (copy(bcpu.Intrinsic) is bcpu.Intrinsic) is True
    assert (copy(bcpu.Intrinsic) is not bcpu.Intrinsic) is False