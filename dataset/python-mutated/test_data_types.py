from zerver.lib.data_types import DictType, EnumType, Equals, ListType, NumberType, OptionalType, StringDictType, TupleType, UnionType, UrlType, schema
from zerver.lib.test_classes import ZulipTestCase

class MiscTest(ZulipTestCase):

    def test_data_type_schema(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        We really only test this to get test coverage.  The\n        code covered here is really only used in testing tools.\n        '
        test_schema = DictType([('type', Equals('realm')), ('maybe_n', OptionalType(int)), ('s', str), ('timestamp', NumberType()), ('flag', bool), ('tup', TupleType([int, str])), ('level', EnumType([1, 2, 3])), ('lst', ListType(int)), ('config', StringDictType(str)), ('value', UnionType([int, str])), ('url', UrlType())])
        expected = "\ntest (dict):\n    config (string_dict):\n        value: str\n    flag: bool\n    level in [1, 2, 3]\n    lst (list):\n        type: int\n    maybe_n: int\n    s: str\n    timestamp: number\n    tup (tuple):\n        0: int\n        1: str\n    type in ['realm']\n    url: str\n    value (union):\n        type: int\n        type: str\n"
        self.assertEqual(schema('test', test_schema).strip(), expected.strip())