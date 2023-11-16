"""
Cryptographically secure random implementation, with fallback on normal random.
"""
import os
import random
import warnings
getrandbits = getattr(random, 'getrandbits', None)
_fromhex = bytes.fromhex

class SecureRandomNotAvailable(RuntimeError):
    """
    Exception raised when no secure random algorithm is found.
    """

class SourceNotAvailable(RuntimeError):
    """
    Internal exception used when a specific random source is not available.
    """

class RandomFactory:
    """
    Factory providing L{secureRandom} and L{insecureRandom} methods.

    You shouldn't have to instantiate this class, use the module level
    functions instead: it is an implementation detail and could be removed or
    changed arbitrarily.
    """
    randomSources = ()
    getrandbits = getrandbits

    def _osUrandom(self, nbytes: int) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper around C{os.urandom} that cleanly manage its absence.\n        '
        try:
            return os.urandom(nbytes)
        except (AttributeError, NotImplementedError) as e:
            raise SourceNotAvailable(e)

    def secureRandom(self, nbytes: int, fallback: bool=False) -> bytes:
        if False:
            return 10
        '\n        Return a number of secure random bytes.\n\n        @param nbytes: number of bytes to generate.\n        @type nbytes: C{int}\n        @param fallback: Whether the function should fallback on non-secure\n            random or not.  Default to C{False}.\n        @type fallback: C{bool}\n\n        @return: a string of random bytes.\n        @rtype: C{str}\n        '
        try:
            return self._osUrandom(nbytes)
        except SourceNotAvailable:
            pass
        if fallback:
            warnings.warn('urandom unavailable - proceeding with non-cryptographically secure random source', category=RuntimeWarning, stacklevel=2)
            return self.insecureRandom(nbytes)
        else:
            raise SecureRandomNotAvailable('No secure random source available')

    def _randBits(self, nbytes: int) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Wrapper around C{os.getrandbits}.\n        '
        if self.getrandbits is not None:
            n = self.getrandbits(nbytes * 8)
            hexBytes = '%%0%dx' % (nbytes * 2) % n
            return _fromhex(hexBytes)
        raise SourceNotAvailable('random.getrandbits is not available')
    _maketrans = bytes.maketrans
    _BYTES = _maketrans(b'', b'')

    def _randModule(self, nbytes: int) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Wrapper around the C{random} module.\n        '
        return b''.join([bytes([random.choice(self._BYTES)]) for i in range(nbytes)])

    def insecureRandom(self, nbytes: int) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Return a number of non secure random bytes.\n\n        @param nbytes: number of bytes to generate.\n        @type nbytes: C{int}\n\n        @return: a string of random bytes.\n        @rtype: C{str}\n        '
        try:
            return self._randBits(nbytes)
        except SourceNotAvailable:
            pass
        return self._randModule(nbytes)
factory = RandomFactory()
secureRandom = factory.secureRandom
insecureRandom = factory.insecureRandom
del factory
__all__ = ['secureRandom', 'insecureRandom', 'SecureRandomNotAvailable']