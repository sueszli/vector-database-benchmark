"""Tests for reaching_definitions module, that only run in Python 3."""
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions_test
from tensorflow.python.platform import test

class ReachingDefinitionsAnalyzerTest(reaching_definitions_test.ReachingDefinitionsAnalyzerTestBase):
    """Tests which can only run in Python 3."""

    def test_nonlocal(self):
        if False:
            for i in range(10):
                print('nop')
        a = 3
        b = 13

        def test_fn():
            if False:
                i = 10
                return i + 15
            nonlocal a
            nonlocal b
            if a:
                b = []
            return (a, b)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[2].test, 1)
        self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[1], 2)
        self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])
        self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

    def test_nonlocal_in_nested_function(self):
        if False:
            i = 10
            return i + 15
        a = 3
        b = 13

        def test_fn():
            if False:
                return 10
            a = 3
            b = 13

            def local_fn():
                if False:
                    i = 10
                    return i + 15
                nonlocal a, b
                if a:
                    b = []
                return (a, b)
            return local_fn()
        node = self._parse_and_analyze(test_fn)
        local_body = node.body[2].body
        self.assertHasDefs(local_body[1].test, 1)
        self.assertHasDefs(local_body[1].body[0].targets[0], 1)
        self.assertHasDefs(local_body[2].value.elts[0], 1)
        self.assertHasDefs(local_body[2].value.elts[1], 2)
        self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])
        self.assertHasDefinedIn(local_body[1], ('a', 'b'))
if __name__ == '__main__':
    test.main()