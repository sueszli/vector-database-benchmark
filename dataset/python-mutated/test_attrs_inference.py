import typing
import attr
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import ResolutionFailed

@attr.s
class Inferrables:
    type_ = attr.ib(type=int)
    type_converter = attr.ib(converter=bool)
    validator_type = attr.ib(validator=attr.validators.instance_of(str))
    validator_type_tuple = attr.ib(validator=attr.validators.instance_of((str, int)))
    validator_type_multiple = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.instance_of((str, int, bool))])
    validator_type_has_overlap = attr.ib(validator=[attr.validators.instance_of(str), attr.validators.instance_of((str, list)), attr.validators.instance_of(object)])
    validator_optional = attr.ib(validator=attr.validators.optional(lambda inst, atrib, val: float(val)))
    validator_in = attr.ib(validator=attr.validators.in_([1, 2, 3]))
    validator_in_multiple = attr.ib(validator=[attr.validators.in_(list(range(100))), attr.validators.in_([1, -1])])
    validator_in_multiple_strings = attr.ib(validator=[attr.validators.in_('abcd'), attr.validators.in_(['ab', 'cd'])])
    typing_list = attr.ib(type=typing.List[int])
    typing_list_of_list = attr.ib(type=typing.List[typing.List[int]])
    typing_dict = attr.ib(type=typing.Dict[str, int])
    typing_optional = attr.ib(type=typing.Optional[bool])
    typing_union = attr.ib(type=typing.Union[str, int])
    has_default = attr.ib(default=0)
    has_default_factory = attr.ib(default=attr.Factory(list))
    has_default_factory_takes_self = attr.ib(default=attr.Factory(lambda _: [], takes_self=True))

@attr.s
class Required:
    a = attr.ib()

@attr.s
class UnhelpfulConverter:
    a = attr.ib(converter=lambda x: x)

@given(st.builds(Inferrables, has_default=..., has_default_factory=...))
def test_attrs_inference_builds(c):
    if False:
        return 10
    pass

@given(st.from_type(Inferrables))
def test_attrs_inference_from_type(c):
    if False:
        print('Hello World!')
    pass

@pytest.mark.parametrize('c', [Required, UnhelpfulConverter])
def test_cannot_infer(c):
    if False:
        while True:
            i = 10
    with pytest.raises(ResolutionFailed):
        st.builds(c).example()

def test_cannot_infer_takes_self():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ResolutionFailed):
        st.builds(Inferrables, has_default_factory_takes_self=...).example()