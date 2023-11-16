from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

def _string_lists(alphabet=None):
    if False:
        for i in range(10):
            print('nop')
    return st.lists(elements=st.text(alphabet=alphabet) if alphabet else st.text(), min_size=0, max_size=3)

class TestStringOps(serial.SerializedTestCase):

    @given(strings=_string_lists())
    @settings(deadline=10000)
    def test_string_prefix(self, strings):
        if False:
            print('Hello World!')
        length = 3
        strings = np.array([a.encode('utf-8') for a in strings], dtype=object)

        def string_prefix_ref(strings):
            if False:
                return 10
            return (np.array([a[:length] for a in strings], dtype=object),)
        op = core.CreateOperator('StringPrefix', ['strings'], ['stripped'], length=length)
        self.assertReferenceChecks(hu.cpu_do, op, [strings], string_prefix_ref)

    @given(strings=_string_lists())
    @settings(deadline=10000)
    def test_string_suffix(self, strings):
        if False:
            return 10
        length = 3
        strings = np.array([a.encode('utf-8') for a in strings], dtype=object)

        def string_suffix_ref(strings):
            if False:
                i = 10
                return i + 15
            return (np.array([a[-length:] for a in strings], dtype=object),)
        op = core.CreateOperator('StringSuffix', ['strings'], ['stripped'], length=length)
        self.assertReferenceChecks(hu.cpu_do, op, [strings], string_suffix_ref)

    @given(strings=st.text(alphabet=['a', 'b']))
    @settings(deadline=10000)
    def test_string_starts_with(self, strings):
        if False:
            print('Hello World!')
        prefix = 'a'
        strings = np.array([str(a) for a in strings], dtype=object)

        def string_starts_with_ref(strings):
            if False:
                while True:
                    i = 10
            return (np.array([a.startswith(prefix) for a in strings], dtype=bool),)
        op = core.CreateOperator('StringStartsWith', ['strings'], ['bools'], prefix=prefix)
        self.assertReferenceChecks(hu.cpu_do, op, [strings], string_starts_with_ref)

    @given(strings=st.text(alphabet=['a', 'b']))
    @settings(deadline=10000)
    def test_string_ends_with(self, strings):
        if False:
            i = 10
            return i + 15
        suffix = 'a'
        strings = np.array([str(a) for a in strings], dtype=object)

        def string_ends_with_ref(strings):
            if False:
                while True:
                    i = 10
            return (np.array([a.endswith(suffix) for a in strings], dtype=bool),)
        op = core.CreateOperator('StringEndsWith', ['strings'], ['bools'], suffix=suffix)
        self.assertReferenceChecks(hu.cpu_do, op, [strings], string_ends_with_ref)

    @given(strings=st.text(alphabet=['a', 'b']))
    @settings(deadline=10000)
    def test_string_equals(self, strings):
        if False:
            while True:
                i = 10
        text = ''
        if strings:
            text = strings[0]
        strings = np.array([str(a) for a in strings], dtype=object)

        def string_equals_ref(strings):
            if False:
                i = 10
                return i + 15
            return (np.array([a == text for a in strings], dtype=bool),)
        op = core.CreateOperator('StringEquals', ['strings'], ['bools'], text=text)
        self.assertReferenceChecks(hu.cpu_do, op, [strings], string_equals_ref)
if __name__ == '__main__':
    import unittest
    unittest.main()