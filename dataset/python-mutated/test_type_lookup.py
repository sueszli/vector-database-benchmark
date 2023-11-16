from typing import Callable
import pytest
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import Concatenate, ParamSpec
from hypothesis.strategies._internal.types import NON_RUNTIME_TYPES
try:
    from typing import TypeGuard
except ImportError:
    TypeGuard = None

@pytest.mark.parametrize('non_runtime_type', NON_RUNTIME_TYPES)
def test_non_runtime_type_cannot_be_resolved(non_runtime_type):
    if False:
        return 10
    strategy = st.from_type(non_runtime_type)
    with pytest.raises(InvalidArgument, match='there is no such thing as a runtime instance'):
        strategy.example()

@pytest.mark.parametrize('non_runtime_type', NON_RUNTIME_TYPES)
def test_non_runtime_type_cannot_be_registered(non_runtime_type):
    if False:
        return 10
    with pytest.raises(InvalidArgument, match='there is no such thing as a runtime instance'):
        st.register_type_strategy(non_runtime_type, st.none())

@pytest.mark.skipif(Concatenate is None, reason='requires python3.10 or higher')
def test_callable_with_concatenate():
    if False:
        for i in range(10):
            print('nop')
    P = ParamSpec('P')
    func_type = Callable[Concatenate[int, P], None]
    strategy = st.from_type(func_type)
    with pytest.raises(InvalidArgument, match="Hypothesis can't yet construct a strategy for instances of a Callable type"):
        strategy.example()
    with pytest.raises(InvalidArgument, match='Cannot register generic type'):
        st.register_type_strategy(func_type, st.none())

@pytest.mark.skipif(ParamSpec is None, reason='requires python3.10 or higher')
def test_callable_with_paramspec():
    if False:
        for i in range(10):
            print('nop')
    P = ParamSpec('P')
    func_type = Callable[P, None]
    strategy = st.from_type(func_type)
    with pytest.raises(InvalidArgument, match="Hypothesis can't yet construct a strategy for instances of a Callable type"):
        strategy.example()
    with pytest.raises(InvalidArgument, match='Cannot register generic type'):
        st.register_type_strategy(func_type, st.none())

@pytest.mark.skipif(TypeGuard is None, reason='requires python3.10 or higher')
def test_callable_return_typegard_type():
    if False:
        return 10
    strategy = st.from_type(Callable[[], TypeGuard[int]])
    with pytest.raises(InvalidArgument, match='Hypothesis cannot yet construct a strategy for callables which are PEP-647 TypeGuards'):
        strategy.example()
    with pytest.raises(InvalidArgument, match='Cannot register generic type'):
        st.register_type_strategy(Callable[[], TypeGuard[int]], st.none())