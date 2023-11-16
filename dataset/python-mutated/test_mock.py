"""Checks that @given, @mock.patch, and pytest fixtures work as expected."""
import math
from unittest import mock
try:
    from pytest import Config
except ImportError:
    from _pytest.config import Config
from hypothesis import given, strategies as st

@given(thing=st.text())
@mock.patch('math.atan')
def test_can_mock_inside_given_without_fixture(atan, thing):
    if False:
        while True:
            i = 10
    assert isinstance(atan, mock.MagicMock)
    assert isinstance(math.atan, mock.MagicMock)

@mock.patch('math.atan')
@given(thing=st.text())
def test_can_mock_outside_given_with_fixture(atan, pytestconfig, thing):
    if False:
        print('Hello World!')
    assert isinstance(atan, mock.MagicMock)
    assert isinstance(math.atan, mock.MagicMock)
    assert isinstance(pytestconfig, Config)

@given(thing=st.text())
def test_can_mock_within_test_with_fixture(pytestconfig, thing):
    if False:
        print('Hello World!')
    assert isinstance(pytestconfig, Config)
    assert not isinstance(math.atan, mock.MagicMock)
    with mock.patch('math.atan') as atan:
        assert isinstance(atan, mock.MagicMock)
        assert isinstance(math.atan, mock.MagicMock)
    assert not isinstance(math.atan, mock.MagicMock)