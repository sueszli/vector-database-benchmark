import os
import pytest
import hypothesis
from hypothesis import errors
from hypothesis.internal import escalation as esc
from hypothesis.internal.compat import BaseExceptionGroup

def test_does_not_escalate_errors_in_non_hypothesis_file():
    if False:
        return 10
    try:
        raise AssertionError
    except AssertionError:
        esc.escalate_hypothesis_internal_error()

def test_does_escalate_errors_in_hypothesis_file(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(esc, 'is_hypothesis_file', lambda x: True)
    with pytest.raises(AssertionError):
        try:
            raise AssertionError
        except AssertionError:
            esc.escalate_hypothesis_internal_error()

def test_does_not_escalate_errors_in_hypothesis_file_if_disabled(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(esc, 'is_hypothesis_file', lambda x: True)
    monkeypatch.setattr(esc, 'PREVENT_ESCALATION', True)
    try:
        raise AssertionError
    except AssertionError:
        esc.escalate_hypothesis_internal_error()

def test_is_hypothesis_file_not_confused_by_prefix(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    root = os.path.dirname(hypothesis.__file__)
    assert esc.is_hypothesis_file(hypothesis.__file__)
    assert esc.is_hypothesis_file(esc.__file__)
    assert not esc.is_hypothesis_file(pytest.__file__)
    assert not esc.is_hypothesis_file(root + '-suffix')
    assert not esc.is_hypothesis_file(root + '-suffix/something.py')

@pytest.mark.parametrize('fname', ['', '<ipython-input-18-f7c304bea5eb>'])
def test_is_hypothesis_file_does_not_error_on_invalid_paths_issue_2319(fname):
    if False:
        i = 10
        return i + 15
    assert not esc.is_hypothesis_file(fname)

def test_multiplefailures_deprecation():
    if False:
        print('Hello World!')
    with pytest.warns(errors.HypothesisDeprecationWarning):
        exc = errors.MultipleFailures
    assert exc is BaseExceptionGroup

def test_errors_attribute_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AttributeError):
        errors.ThisIsNotARealAttributeDontCreateSomethingWithThisName

def test_handles_null_traceback():
    if False:
        print('Hello World!')
    esc.get_interesting_origin(Exception())