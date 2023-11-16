from __future__ import annotations
import pytest
from ansible.module_utils.parsing.convert_bool import boolean

class TestBoolean:

    def test_bools(self):
        if False:
            print('Hello World!')
        assert boolean(True) is True
        assert boolean(False) is False

    def test_none(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            assert boolean(None, strict=True) is False
        assert boolean(None, strict=False) is False

    def test_numbers(self):
        if False:
            return 10
        assert boolean(1) is True
        assert boolean(0) is False
        assert boolean(0.0) is False

    def test_strings(self):
        if False:
            for i in range(10):
                print('nop')
        assert boolean('true') is True
        assert boolean('TRUE') is True
        assert boolean('t') is True
        assert boolean('yes') is True
        assert boolean('y') is True
        assert boolean('on') is True

    def test_junk_values_nonstrict(self):
        if False:
            while True:
                i = 10
        assert boolean('flibbity', strict=False) is False
        assert boolean(42, strict=False) is False
        assert boolean(42.0, strict=False) is False
        assert boolean(object(), strict=False) is False

    def test_junk_values_strict(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            assert boolean('flibbity', strict=True) is False
        with pytest.raises(TypeError):
            assert boolean(42, strict=True) is False
        with pytest.raises(TypeError):
            assert boolean(42.0, strict=True) is False
        with pytest.raises(TypeError):
            assert boolean(object(), strict=True) is False