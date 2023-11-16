"""
Tests for object finalization semantics, as outlined in PEP 442.
"""
import contextlib
import gc
import unittest
import weakref
try:
    from _testcapi import with_tp_del
except ImportError:

    def with_tp_del(cls):
        if False:
            print('Hello World!')

        class C(object):

            def __new__(cls, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                raise TypeError('requires _testcapi.with_tp_del')
        return C
try:
    from _testcapi import without_gc
except ImportError:

    def without_gc(cls):
        if False:
            print('Hello World!')

        class C:

            def __new__(cls, *args, **kwargs):
                if False:
                    return 10
                raise TypeError('requires _testcapi.without_gc')
        return C
from test import support

class NonGCSimpleBase:
    """
    The base class for all the objects under test, equipped with various
    testing features.
    """
    survivors = []
    del_calls = []
    tp_del_calls = []
    errors = []
    _cleaning = False
    __slots__ = ()

    @classmethod
    def _cleanup(cls):
        if False:
            return 10
        cls.survivors.clear()
        cls.errors.clear()
        gc.garbage.clear()
        gc.collect()
        cls.del_calls.clear()
        cls.tp_del_calls.clear()

    @classmethod
    @contextlib.contextmanager
    def test(cls):
        if False:
            i = 10
            return i + 15
        '\n        A context manager to use around all finalization tests.\n        '
        with support.disable_gc():
            cls.del_calls.clear()
            cls.tp_del_calls.clear()
            NonGCSimpleBase._cleaning = False
            try:
                yield
                if cls.errors:
                    raise cls.errors[0]
            finally:
                NonGCSimpleBase._cleaning = True
                cls._cleanup()

    def check_sanity(self):
        if False:
            return 10
        '\n        Check the object is sane (non-broken).\n        '

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        PEP 442 finalizer.  Record that this was called, check the\n        object is in a sane state, and invoke a side effect.\n        '
        try:
            if not self._cleaning:
                self.del_calls.append(id(self))
                self.check_sanity()
                self.side_effect()
        except Exception as e:
            self.errors.append(e)

    def side_effect(self):
        if False:
            print('Hello World!')
        '\n        A side effect called on destruction.\n        '

class SimpleBase(NonGCSimpleBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.id_ = id(self)

    def check_sanity(self):
        if False:
            while True:
                i = 10
        assert self.id_ == id(self)

@without_gc
class NonGC(NonGCSimpleBase):
    __slots__ = ()

@without_gc
class NonGCResurrector(NonGCSimpleBase):
    __slots__ = ()

    def side_effect(self):
        if False:
            while True:
                i = 10
        '\n        Resurrect self by storing self in a class-wide list.\n        '
        self.survivors.append(self)

class Simple(SimpleBase):
    pass

class SimpleResurrector(SimpleBase):

    def side_effect(self):
        if False:
            print('Hello World!')
        '\n        Resurrect self by storing self in a class-wide list.\n        '
        self.survivors.append(self)

class TestBase:

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.old_garbage = gc.garbage[:]
        gc.garbage[:] = []

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.assertEqual(gc.garbage, [])
        finally:
            del self.old_garbage
            gc.collect()

    def assert_del_calls(self, ids):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(sorted(SimpleBase.del_calls), sorted(ids))

    def assert_tp_del_calls(self, ids):
        if False:
            i = 10
            return i + 15
        self.assertEqual(sorted(SimpleBase.tp_del_calls), sorted(ids))

    def assert_survivors(self, ids):
        if False:
            i = 10
            return i + 15
        self.assertEqual(sorted((id(x) for x in SimpleBase.survivors)), sorted(ids))

    def assert_garbage(self, ids):
        if False:
            while True:
                i = 10
        self.assertEqual(sorted((id(x) for x in gc.garbage)), sorted(ids))

    def clear_survivors(self):
        if False:
            for i in range(10):
                print('nop')
        SimpleBase.survivors.clear()

class SimpleFinalizationTest(TestBase, unittest.TestCase):
    """
    Test finalization without refcycles.
    """

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = Simple()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])

    def test_simple_resurrect(self):
        if False:
            while True:
                i = 10
        with SimpleBase.test():
            s = SimpleResurrector()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors(ids)
            self.assertIsNot(wr(), None)
            self.clear_survivors()
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
        self.assertIs(wr(), None)

    @support.cpython_only
    def test_non_gc(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = NonGC()
            self.assertFalse(gc.is_tracked(s))
            ids = [id(s)]
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])

    @support.cpython_only
    def test_non_gc_resurrect(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = NonGCResurrector()
            self.assertFalse(gc.is_tracked(s))
            ids = [id(s)]
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors(ids)
            self.clear_survivors()
            gc.collect()
            self.assert_del_calls(ids * 2)
            self.assert_survivors(ids)

class SelfCycleBase:

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.ref = self

    def check_sanity(self):
        if False:
            while True:
                i = 10
        super().check_sanity()
        assert self.ref is self

class SimpleSelfCycle(SelfCycleBase, Simple):
    pass

class SelfCycleResurrector(SelfCycleBase, SimpleResurrector):
    pass

class SuicidalSelfCycle(SelfCycleBase, Simple):

    def side_effect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Explicitly break the reference cycle.\n        '
        self.ref = None

class SelfCycleFinalizationTest(TestBase, unittest.TestCase):
    """
    Test finalization of an object having a single cyclic reference to
    itself.
    """

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = SimpleSelfCycle()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])

    def test_simple_resurrect(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = SelfCycleResurrector()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors(ids)
            self.assertIs(wr(), None)
            self.clear_survivors()
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)

    def test_simple_suicide(self):
        if False:
            for i in range(10):
                print('nop')
        with SimpleBase.test():
            s = SuicidalSelfCycle()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)

class ChainedBase:

    def chain(self, left):
        if False:
            for i in range(10):
                print('nop')
        self.suicided = False
        self.left = left
        left.right = self

    def check_sanity(self):
        if False:
            while True:
                i = 10
        super().check_sanity()
        if self.suicided:
            assert self.left is None
            assert self.right is None
        else:
            left = self.left
            if left.suicided:
                assert left.right is None
            else:
                assert left.right is self
            right = self.right
            if right.suicided:
                assert right.left is None
            else:
                assert right.left is self

class SimpleChained(ChainedBase, Simple):
    pass

class ChainedResurrector(ChainedBase, SimpleResurrector):
    pass

class SuicidalChained(ChainedBase, Simple):

    def side_effect(self):
        if False:
            while True:
                i = 10
        '\n        Explicitly break the reference cycle.\n        '
        self.suicided = True
        self.left = None
        self.right = None

class CycleChainFinalizationTest(TestBase, unittest.TestCase):
    """
    Test finalization of a cyclic chain.  These tests are similar in
    spirit to the self-cycle tests above, but the collectable object
    graph isn't trivial anymore.
    """

    def build_chain(self, classes):
        if False:
            return 10
        nodes = [cls() for cls in classes]
        for i in range(len(nodes)):
            nodes[i].chain(nodes[i - 1])
        return nodes

    def check_non_resurrecting_chain(self, classes):
        if False:
            return 10
        N = len(classes)
        with SimpleBase.test():
            nodes = self.build_chain(classes)
            ids = [id(s) for s in nodes]
            wrs = [weakref.ref(s) for s in nodes]
            del nodes
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])
            self.assertEqual([wr() for wr in wrs], [None] * N)
            gc.collect()
            self.assert_del_calls(ids)

    def check_resurrecting_chain(self, classes):
        if False:
            i = 10
            return i + 15
        N = len(classes)
        with SimpleBase.test():
            nodes = self.build_chain(classes)
            N = len(nodes)
            ids = [id(s) for s in nodes]
            survivor_ids = [id(s) for s in nodes if isinstance(s, SimpleResurrector)]
            wrs = [weakref.ref(s) for s in nodes]
            del nodes
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors(survivor_ids)
            self.assertEqual([wr() for wr in wrs], [None] * N)
            self.clear_survivors()
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_survivors([])

    def test_homogenous(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_non_resurrecting_chain([SimpleChained] * 3)

    def test_homogenous_resurrect(self):
        if False:
            i = 10
            return i + 15
        self.check_resurrecting_chain([ChainedResurrector] * 3)

    def test_homogenous_suicidal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_non_resurrecting_chain([SuicidalChained] * 3)

    def test_heterogenous_suicidal_one(self):
        if False:
            return 10
        self.check_non_resurrecting_chain([SuicidalChained, SimpleChained] * 2)

    def test_heterogenous_suicidal_two(self):
        if False:
            print('Hello World!')
        self.check_non_resurrecting_chain([SuicidalChained] * 2 + [SimpleChained] * 2)

    def test_heterogenous_resurrect_one(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_resurrecting_chain([ChainedResurrector, SimpleChained] * 2)

    def test_heterogenous_resurrect_two(self):
        if False:
            i = 10
            return i + 15
        self.check_resurrecting_chain([ChainedResurrector, SimpleChained, SuicidalChained] * 2)

    def test_heterogenous_resurrect_three(self):
        if False:
            while True:
                i = 10
        self.check_resurrecting_chain([ChainedResurrector] * 2 + [SimpleChained] * 2 + [SuicidalChained] * 2)

class LegacyBase(SimpleBase):

    def __del__(self):
        if False:
            i = 10
            return i + 15
        try:
            if not self._cleaning:
                self.del_calls.append(id(self))
                self.check_sanity()
        except Exception as e:
            self.errors.append(e)

    def __tp_del__(self):
        if False:
            while True:
                i = 10
        '\n        Legacy (pre-PEP 442) finalizer, mapped to a tp_del slot.\n        '
        try:
            if not self._cleaning:
                self.tp_del_calls.append(id(self))
                self.check_sanity()
                self.side_effect()
        except Exception as e:
            self.errors.append(e)

@with_tp_del
class Legacy(LegacyBase):
    pass

@with_tp_del
class LegacyResurrector(LegacyBase):

    def side_effect(self):
        if False:
            i = 10
            return i + 15
        '\n        Resurrect self by storing self in a class-wide list.\n        '
        self.survivors.append(self)

@with_tp_del
class LegacySelfCycle(SelfCycleBase, LegacyBase):
    pass

@support.cpython_only
class LegacyFinalizationTest(TestBase, unittest.TestCase):
    """
    Test finalization of objects with a tp_del.
    """

    def tearDown(self):
        if False:
            while True:
                i = 10
        gc.garbage.clear()
        gc.collect()
        super().tearDown()

    def test_legacy(self):
        if False:
            while True:
                i = 10
        with SimpleBase.test():
            s = Legacy()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_tp_del_calls(ids)
            self.assert_survivors([])
            self.assertIs(wr(), None)
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_tp_del_calls(ids)

    def test_legacy_resurrect(self):
        if False:
            i = 10
            return i + 15
        with SimpleBase.test():
            s = LegacyResurrector()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_tp_del_calls(ids)
            self.assert_survivors(ids)
            self.assertIs(wr(), None)
            self.clear_survivors()
            gc.collect()
            self.assert_del_calls(ids)
            self.assert_tp_del_calls(ids * 2)
            self.assert_survivors(ids)
        self.assertIs(wr(), None)

    def test_legacy_self_cycle(self):
        if False:
            while True:
                i = 10
        with SimpleBase.test():
            s = LegacySelfCycle()
            ids = [id(s)]
            wr = weakref.ref(s)
            del s
            gc.collect()
            self.assert_del_calls([])
            self.assert_tp_del_calls([])
            self.assert_survivors([])
            self.assert_garbage(ids)
            self.assertIsNot(wr(), None)
            gc.garbage[0].ref = None
        self.assert_garbage([])
        self.assertIs(wr(), None)
if __name__ == '__main__':
    unittest.main()