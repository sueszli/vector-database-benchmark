"""
Implement utils for supporting retargeting of dispatchers.

WARNING: Features defined in this file are experimental. The API may change
         without notice.
"""
import abc
import weakref
from numba.core import errors

class RetargetCache:
    """Cache for retargeted dispatchers.

    The cache uses the original dispatcher as the key.
    """
    container_type = weakref.WeakKeyDictionary

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cache = self.container_type()
        self._stat_hit = 0
        self._stat_miss = 0

    def save_cache(self, orig_disp, new_disp):
        if False:
            for i in range(10):
                print('nop')
        'Save a dispatcher associated with the given key.\n        '
        self._cache[orig_disp] = new_disp

    def load_cache(self, orig_disp):
        if False:
            for i in range(10):
                print('nop')
        'Load a dispatcher associated with the given key.\n        '
        out = self._cache.get(orig_disp)
        if out is None:
            self._stat_miss += 1
        else:
            self._stat_hit += 1
        return out

    def items(self):
        if False:
            i = 10
            return i + 15
        'Returns the contents of the cache.\n        '
        return self._cache.items()

    def stats(self):
        if False:
            i = 10
            return i + 15
        'Returns stats regarding cache hit/miss.\n        '
        return {'hit': self._stat_hit, 'miss': self._stat_miss}

class BaseRetarget(abc.ABC):
    """Abstract base class for retargeting logic.
    """

    @abc.abstractmethod
    def check_compatible(self, orig_disp):
        if False:
            for i in range(10):
                print('nop')
        'Check that the retarget is compatible.\n\n        This method does not return anything meaningful (e.g. None)\n        Incompatibility is signalled via raising an exception.\n        '
        pass

    @abc.abstractmethod
    def retarget(self, orig_disp):
        if False:
            print('Hello World!')
        'Retargets the given dispatcher and returns a new dispatcher-like\n        callable. Or, returns the original dispatcher if the the target_backend\n        will not change.\n        '
        pass

class BasicRetarget(BaseRetarget):
    """A basic retargeting implementation for a single output target.

    This class has two abstract methods/properties that subclasses must define.

    - `output_target` must return output target name.
    - `compile_retarget` must define the logic to retarget the given dispatcher.

    By default, this class uses `RetargetCache` as the internal cache. This
    can be modified by overriding the `.cache_type` class attribute.

    """
    cache_type = RetargetCache

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.cache = self.cache_type()

    @abc.abstractproperty
    def output_target(self) -> str:
        if False:
            return 10
        'Returns the output target name.\n\n        See numba/tests/test_retargeting.py for example usage.\n        '
        pass

    @abc.abstractmethod
    def compile_retarget(self, orig_disp):
        if False:
            return 10
        'Returns the retargeted dispatcher.\n\n        See numba/tests/test_retargeting.py for example usage.\n        '
        pass

    def check_compatible(self, orig_disp):
        if False:
            while True:
                i = 10
        '\n        This implementation checks that\n        `self.output_target == orig_disp._required_target_backend`\n        '
        required_target = orig_disp._required_target_backend
        output_target = self.output_target
        if required_target is not None:
            if output_target != required_target:
                m = f'The output target does match the required target: {output_target} != {required_target}.'
                raise errors.CompilerError(m)

    def retarget(self, orig_disp):
        if False:
            print('Hello World!')
        'Apply retargeting to orig_disp.\n\n        The retargeted dispatchers are cached for future use.\n        '
        cache = self.cache
        opts = orig_disp.targetoptions
        if opts.get('target_backend') == self.output_target:
            return orig_disp
        cached = cache.load_cache(orig_disp)
        if cached is None:
            out = self.compile_retarget(orig_disp)
            cache.save_cache(orig_disp, out)
        else:
            out = cached
        return out