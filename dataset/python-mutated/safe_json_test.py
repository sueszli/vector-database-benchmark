import unittest
from io import StringIO
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import typing_extensions
from .. import safe_json
T = TypeVar('T')

class Movie(typing_extensions.TypedDict):
    name: str
    year: int

class MovieWithRating(Movie):
    rating: float

class MovieWithArbitraryDictionary(Movie):
    dictionary: Dict[str, Any]

class MovieWithUnion(Movie):
    int_or_str: Union[int, str]

class MovieWithNonRequiredField(Movie, total=False):
    not_required: str

class MovieAlternative(typing_extensions.TypedDict):
    name: str
    year: int

class BasicTestCase(unittest.TestCase):

    def _assert_loads(self, input: str, target_type: Type[T], output: T) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(safe_json.loads(input, target_type), output)

    def _assert_loads_fails(self, input: str, target_type: Type[T]) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads(input, target_type)

    def test_loads(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(safe_json.loads('1', int), 1)
        self.assertEqual(safe_json.loads('true', bool), True)
        self.assertEqual(safe_json.loads('1.1', float), 1.1)
        self.assertEqual(safe_json.loads('"string"', str), 'string')
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1', bool)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1', float)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1', str)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('true', float)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('true', str)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1.1', int)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1.1', bool)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('1.1', str)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('hello', int)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('hello', bool)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('hello', float)
        self.assertEqual(safe_json.loads('[]', List[int]), [])
        self.assertEqual(safe_json.loads('[1]', List[int]), [1])
        self.assertEqual(safe_json.loads('[1, 2]', List[int]), [1, 2])
        self.assertEqual(safe_json.loads('[{"1": 1}]', List[Dict[str, int]]), [{'1': 1}])
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads("[1, 'string']", List[int])
        self.assertEqual(safe_json.loads('{}', Dict[int, str]), {})
        self.assertEqual(safe_json.loads('{"1": 1}', Dict[str, int]), {'1': 1})
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('{"1": "string"}', Dict[str, int])
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('{"1": 1, "2": "2"}', Dict[str, int])
        self.assertEqual(safe_json.loads('{"1": {"2": 3}}', Dict[str, Dict[str, int]]), {'1': {'2': 3}})
        self.assertEqual(safe_json.loads('{"name": "The Matrix", "year": 1999}', Movie), {'name': 'The Matrix', 'year': 1999})
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.loads('{"name": "The Matrix", "year": ""}', Movie)
        self.assertEqual(safe_json.loads('[1]', List[Any]), [1])
        self.assertEqual(safe_json.loads('[{"1": 1}]', List[Any]), [{'1': 1}])
        self.assertEqual(safe_json.loads('[1]', List[Optional[int]]), [1])
        self.assertEqual(safe_json.loads('[null, 2]', List[Optional[int]]), [None, 2])
        self.assertEqual(safe_json.loads('[1]', List[str], validate=False), [1])

    def test_validate(self) -> None:
        if False:
            while True:
                i = 10
        parsedListStr = ['1', '2']
        self.assertEqual(safe_json.validate(parsedListStr, List[str]), parsedListStr)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.validate(parsedListStr, List[int])
        parsedDictBasic = {'1': 1}
        self.assertEqual(safe_json.validate(parsedDictBasic, Dict[str, int]), parsedDictBasic)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.validate(parsedDictBasic, List[Any])
        parsedDictNested = {'1': {'2': 3}}
        self.assertEqual(safe_json.validate(parsedDictNested, Dict[str, Dict[str, int]]), parsedDictNested)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.validate(parsedDictNested, Dict[str, int])
        parsedDictTyped = {'name': 'The Matrix', 'year': 1999}
        parsedDictTypedFailing = {'name': 'The Matrix', 'year': ''}
        self.assertEqual(safe_json.validate(parsedDictTyped, Movie), parsedDictTyped)
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.validate(parsedDictTypedFailing, Movie)
        parsedAny = [{'1': 1}]
        self.assertEqual(safe_json.validate(parsedAny, List[Any]), parsedAny)
        parsedOptionals = [2, None, 4]
        self.assertEqual(safe_json.validate(parsedOptionals, List[Optional[int]]), parsedOptionals)

    def test_load(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        f = StringIO('{"1": {"2": 3}}')
        self.assertEqual(safe_json.load(f, Dict[str, Dict[str, int]]), {'1': {'2': 3}})
        with self.assertRaises(safe_json.InvalidJson):
            safe_json.load(f, Dict[int, Dict[int, int]])

    def test_loads_typed_dictionary(self) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_loads('{"name": "The Matrix Reloaded", "year": 1999, "extra_field": "hello"}', Movie, {'name': 'The Matrix Reloaded', 'year': 1999, 'extra_field': 'hello'})
        self._assert_loads('{"name": "The Matrix", "year": 1999, "rating": 9.0}', MovieWithRating, {'name': 'The Matrix', 'year': 1999, 'rating': 9.0})
        self._assert_loads_fails('{"name": "The Matrix", "year": 1999, "rating": "not a float"}', MovieWithRating)
        self._assert_loads('{"name": "The Matrix", "year": 1999,' + ' "dictionary": {"foo": "bar", "baz": {}}}', MovieWithArbitraryDictionary, {'name': 'The Matrix', 'year': 1999, 'dictionary': {'foo': 'bar', 'baz': {}}})
        self._assert_loads_fails('{"name": "The Matrix", "year": 1999, "dictionary": [1, 2]}', MovieWithArbitraryDictionary)
        self._assert_loads_fails('{"name": "The Matrix", "year": 1999, "int_or_str": 1}', MovieWithUnion)
        self._assert_loads('{"name": "The Matrix", "year": 1999, "not_required": "hello"}', MovieWithNonRequiredField, {'name': 'The Matrix', 'year': 1999, 'not_required': 'hello'})
        self._assert_loads_fails('{"name": "The Matrix", "year": 1999}', MovieWithNonRequiredField)
        self._assert_loads('{"name": "The Matrix", "year": 1999}', MovieAlternative, {'name': 'The Matrix', 'year': 1999})
if __name__ == '__main__':
    unittest.main()