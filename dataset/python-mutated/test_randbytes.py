"""
Test cases for L{twisted.python.randbytes}.
"""
from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest

class _SupportsAssertions(Protocol):

    def assertEqual(self, a: object, b: object) -> object:
        if False:
            print('Hello World!')
        ...

    def assertNotEqual(self, a: object, b: object) -> object:
        if False:
            i = 10
            return i + 15
        ...

class SecureRandomTestCaseBase:
    """
    Base class for secureRandom test cases.
    """

    def _check(self: _SupportsAssertions, source: Callable[[int], bytes]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        The given random bytes source should return the number of bytes\n        requested each time it is called and should probably not return the\n        same bytes on two consecutive calls (although this is a perfectly\n        legitimate occurrence and rejecting it may generate a spurious failure\n        -- maybe we'll get lucky and the heat death with come first).\n        "
        for nbytes in range(17, 25):
            s = source(nbytes)
            self.assertEqual(len(s), nbytes)
            s2 = source(nbytes)
            self.assertEqual(len(s2), nbytes)
            self.assertNotEqual(s2, s)

class SecureRandomTests(SecureRandomTestCaseBase, unittest.TestCase):
    """
    Test secureRandom under normal conditions.
    """

    def test_normal(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{randbytes.secureRandom} should return a string of the requested\n        length and make some effort to make its result otherwise unpredictable.\n        '
        self._check(randbytes.secureRandom)

class ConditionalSecureRandomTests(SecureRandomTestCaseBase, unittest.SynchronousTestCase):
    """
    Test random sources one by one, then remove it to.
    """

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        '\n        Create a L{randbytes.RandomFactory} to use in the tests.\n        '
        self.factory = randbytes.RandomFactory()

    def errorFactory(self, nbytes: object) -> NoReturn:
        if False:
            while True:
                i = 10
        '\n        A factory raising an error when a source is not available.\n        '
        raise randbytes.SourceNotAvailable()

    def test_osUrandom(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{RandomFactory._osUrandom} should work as a random source whenever\n        L{os.urandom} is available.\n        '
        self._check(self.factory._osUrandom)

    def test_withoutAnything(self) -> None:
        if False:
            return 10
        '\n        Remove all secure sources and assert it raises a failure. Then try the\n        fallback parameter.\n        '
        self.factory._osUrandom = self.errorFactory
        self.assertRaises(randbytes.SecureRandomNotAvailable, self.factory.secureRandom, 18)

        def wrapper() -> bytes:
            if False:
                print('Hello World!')
            return self.factory.secureRandom(18, fallback=True)
        s = self.assertWarns(RuntimeWarning, 'urandom unavailable - proceeding with non-cryptographically secure random source', __file__, wrapper)
        self.assertEqual(len(s), 18)

class RandomBaseTests(SecureRandomTestCaseBase, unittest.SynchronousTestCase):
    """
    'Normal' random test cases.
    """

    def test_normal(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test basic case.\n        '
        self._check(randbytes.insecureRandom)

    def test_withoutGetrandbits(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test C{insecureRandom} without C{random.getrandbits}.\n        '
        factory = randbytes.RandomFactory()
        factory.getrandbits = None
        self._check(factory.insecureRandom)