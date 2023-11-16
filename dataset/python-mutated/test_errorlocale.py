from __future__ import annotations
from copy import deepcopy
from typing import Any
from unittest import TestCase
from sentry.lang.javascript.errorlocale import translate_exception, translate_message

class ErrorLocaleTest(TestCase):

    def test_basic_translation(self):
        if False:
            while True:
                i = 10
        actual = 'Type mismatch'
        expected = translate_message('Typenkonflikt')
        assert actual == expected

    def test_unicode_translation(self):
        if False:
            print('Hello World!')
        expected = 'Division by zero'
        actual = translate_message('División por cero')
        assert actual == expected

    def test_same_translation(self):
        if False:
            return 10
        expected = 'Out of memory'
        actual = translate_message('Out of memory')
        assert actual == expected

    def test_unknown_translation(self):
        if False:
            print('Hello World!')
        expected = 'Some unknown message'
        actual = translate_message('Some unknown message')
        assert actual == expected

    def test_translation_with_type(self):
        if False:
            while True:
                i = 10
        expected = 'RangeError: Subscript out of range'
        actual = translate_message('RangeError: Indeks poza zakresem')
        assert actual == expected

    def test_translation_with_type_and_colon(self):
        if False:
            return 10
        expected = 'RangeError: Cannot define property: object is not extensible'
        actual = translate_message('RangeError: Nie można zdefiniować właściwości: obiekt nie jest rozszerzalny')
        assert actual == expected

    def test_interpolated_translation(self):
        if False:
            i = 10
            return i + 15
        expected = "Type 'foo' not found"
        actual = translate_message('Nie odnaleziono typu „foo”')
        assert actual == expected

    def test_interpolated_translation_with_colon(self):
        if False:
            while True:
                i = 10
        expected = "'this' is not of expected type: foo"
        actual = translate_message('Typ obiektu „this” jest inny niż oczekiwany: foo')
        assert actual == expected

    def test_interpolated_translation_with_colon_in_front(self):
        if False:
            return 10
        expected = 'foo: an unexpected failure occurred while trying to obtain metadata information'
        actual = translate_message('foo: wystąpił nieoczekiwany błąd podczas próby uzyskania informacji o metadanych')
        assert actual == expected

    def test_interpolated_translation_with_type(self):
        if False:
            for i in range(10):
                print('nop')
        expected = "TypeError: Type 'foo' not found"
        actual = translate_message('TypeError: Nie odnaleziono typu „foo”')
        assert actual == expected

    def test_interpolated_translation_with_type_and_colon(self):
        if False:
            while True:
                i = 10
        expected = "ReferenceError: Cannot modify property 'foo': 'length' is not writable"
        actual = translate_message('ReferenceError: Nie można zmodyfikować właściwości „foo”: wartość „length” jest niezapisywalna')
        assert actual == expected

    def test_translate_exception(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'logentry': {'message': 'Typenkonflikt', 'formatted': 'Typenkonflikt'}, 'exception': {'values': [{'value': 'Typenkonflikt'}, {'value': 'Typenkonflikt'}]}}
        translate_exception(data)
        assert data == {'logentry': {'message': 'Type mismatch', 'formatted': 'Type mismatch'}, 'exception': {'values': [{'value': 'Type mismatch'}, {'value': 'Type mismatch'}]}}

    def test_translate_exception_missing(self):
        if False:
            i = 10
            return i + 15
        data: dict[str, Any] = {}
        translate_exception(data)
        assert data == {}

    def test_translate_exception_none(self):
        if False:
            print('Hello World!')
        expected = {'logentry': {'message': None, 'formatted': None}, 'exception': {'values': [None, {'value': None}]}}
        actual = deepcopy(expected)
        translate_exception(actual)
        assert actual == expected