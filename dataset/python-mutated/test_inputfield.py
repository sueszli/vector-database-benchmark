from functools import partial
from pytest import raises
from ..inputfield import InputField
from ..structures import NonNull
from .utils import MyLazyType

def test_inputfield_required():
    if False:
        for i in range(10):
            print('nop')
    MyType = object()
    field = InputField(MyType, required=True)
    assert isinstance(field.type, NonNull)
    assert field.type.of_type == MyType

def test_inputfield_deprecated():
    if False:
        return 10
    MyType = object()
    deprecation_reason = 'deprecated'
    field = InputField(MyType, required=False, deprecation_reason=deprecation_reason)
    assert isinstance(field.type, type(MyType))
    assert field.deprecation_reason == deprecation_reason

def test_inputfield_required_deprecated():
    if False:
        i = 10
        return i + 15
    MyType = object()
    with raises(AssertionError) as exc_info:
        InputField(MyType, name='input', required=True, deprecation_reason='deprecated')
    assert str(exc_info.value) == 'InputField input is required, cannot deprecate it.'

def test_inputfield_with_lazy_type():
    if False:
        print('Hello World!')
    MyType = object()
    field = InputField(lambda : MyType)
    assert field.type == MyType

def test_inputfield_with_lazy_partial_type():
    if False:
        return 10
    MyType = object()
    field = InputField(partial(lambda : MyType))
    assert field.type == MyType

def test_inputfield_with_string_type():
    if False:
        return 10
    field = InputField('graphene.types.tests.utils.MyLazyType')
    assert field.type == MyLazyType