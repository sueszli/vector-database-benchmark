"""Tests for SharedVariableCreator."""
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1

class CanonicalizeVariableNameTest(test.TestCase):

    def _canonicalize(self, name):
        if False:
            for i in range(10):
                print('nop')
        return shared_variable_creator._canonicalize_variable_name(name)

    def testNoName(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('Variable', self._canonicalize(None))

    def testPatternInMiddle(self):
        if False:
            while True:
                i = 10
        self.assertEqual('foo/bar/baz', self._canonicalize('foo_1/bar_1/baz'))

    def testPatternAtEnd(self):
        if False:
            return 10
        self.assertEqual('foo', self._canonicalize('foo_1'))

    def testWrongPatterns(self):
        if False:
            return 10
        self.assertEqual('foo_1:0', self._canonicalize('foo_1:0'))
        self.assertEqual('foo1', self._canonicalize('foo1'))
        self.assertEqual('foo_a', self._canonicalize('foo_a'))

class SharedVariableCreatorTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testSharedVariable(self):
        if False:
            i = 10
            return i + 15
        shared_variable_store = {}
        num_devices = 3
        creator_fns = []
        for i in range(num_devices):
            creator_fn = shared_variable_creator.make_fn(shared_variable_store, i)
            creator_fns.append(creator_fn)
        with variable_scope.variable_creator_scope(creator_fns[0]):
            v0 = variable_v1.VariableV1(1.0, name='foo')
        with variable_scope.variable_creator_scope(creator_fns[1]):
            v1 = variable_v1.VariableV1(1.0, name='foo')
        with variable_scope.variable_creator_scope(creator_fns[2]):
            v2 = variable_v1.VariableV1(1.0, name='foo')
        self.assertIs(v1, v0)
        self.assertIs(v2, v0)
if __name__ == '__main__':
    test.main()