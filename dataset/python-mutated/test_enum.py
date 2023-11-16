from enum import Enum, auto
import pytest
from faker.providers.python import EmptyEnumException

class _TestEnumWithNoElements(Enum):
    pass

class _TestEnumWithSingleElement(Enum):
    Single = auto

class _TestEnum(Enum):
    A = auto
    B = auto
    C = auto

class TestEnumProvider:
    num_samples = 100

    def test_enum(self, faker, num_samples):
        if False:
            while True:
                i = 10
        for _ in range(num_samples):
            actual = faker.enum(_TestEnum)
            assert actual in (_TestEnum.A, _TestEnum.B, _TestEnum.C)

    def test_enum_single(self, faker):
        if False:
            return 10
        assert faker.enum(_TestEnumWithSingleElement) == _TestEnumWithSingleElement.Single
        assert faker.enum(_TestEnumWithSingleElement) == _TestEnumWithSingleElement.Single

    def test_empty_enum_raises(self, faker):
        if False:
            print('Hello World!')
        with pytest.raises(EmptyEnumException, match="The provided Enum: '_TestEnumWithNoElements' has no members."):
            faker.enum(_TestEnumWithNoElements)

    def test_none_raises(self, faker):
        if False:
            return 10
        with pytest.raises(ValueError):
            faker.enum(None)

    def test_incorrect_type_raises(self, faker):
        if False:
            print('Hello World!')
        not_an_enum_type = type('NotAnEnumType')
        with pytest.raises(TypeError):
            faker.enum(not_an_enum_type)