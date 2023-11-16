import sys
import unittest
from vimspector import core_utils

class TestOverrideDict(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

    def test_override(self):
        if False:
            i = 10
            return i + 15
        tests = (({}, ({}, {})), ({'a': 'a'}, ({'a': 'a'}, {})), ({'a': 'a'}, ({}, {'a': 'a'})), ({'a': 'a', 'b': {'a': 'aa'}}, ({'a': 'a'}, {'b': {'a': 'aa'}})), ({'outer': {'inner': {'key': 'newValue', 'existingKey': True}}, 'newKey': {'newDict': True}}, ({'outer': {'inner': {'key': 'oldValue', 'existingKey': True}}}, {'outer': {'inner': {'key': 'newValue'}}, 'newKey': {'newDict': True}})), ({'outer': {'inner': {'key': 'newValue'}}, 'newKey': {'newDict': True}}, ({'outer': {'inner': {'key': 'oldValue', 'existingKey': True}}}, {'outer': {'inner': {'key': 'newValue', '!existingKey': 'REMOVE'}}, 'newKey': {'newDict': True}})))
        for (expect, t) in tests:
            with self.subTest(t[0]):
                self.assertDictEqual(expect, core_utils.override(*t))
assert unittest.main(module=__name__, testRunner=unittest.TextTestRunner(sys.stdout), exit=False).result.wasSuccessful()