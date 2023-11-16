import unittest
from coalib.bears.Bear import Bear
from coalib.collecting import Dependencies
from coalib.core.CircularDependencyError import CircularDependencyError

class ResolvableBear1(Bear):
    BEAR_DEPS = {Bear}

class ResolvableBear2(Bear):
    BEAR_DEPS = {ResolvableBear1, Bear}

class UnresolvableBear1(Bear):
    BEAR_DEPS = {ResolvableBear1, Bear}

class UnresolvableBear2(Bear):
    BEAR_DEPS = {ResolvableBear1, Bear, UnresolvableBear1}

class UnresolvableBear3(Bear):
    BEAR_DEPS = {ResolvableBear1, Bear, UnresolvableBear2}

class DependenciesTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        setattr(UnresolvableBear1, 'BEAR_DEPS', {ResolvableBear1, Bear, UnresolvableBear3})

    def test_no_deps(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(Dependencies.resolve([Bear, Bear])), 1)

    def test_resolvable_deps(self):
        if False:
            return 10
        self.assertEqual(Dependencies.resolve([ResolvableBear1, ResolvableBear2]), [Bear, ResolvableBear1, ResolvableBear2])

    def test_unresolvable_deps(self):
        if False:
            return 10
        self.assertRaises(CircularDependencyError, Dependencies.resolve, [UnresolvableBear1])