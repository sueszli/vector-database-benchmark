"""Inter-object utility class."""
from __future__ import absolute_import
from bzrlib.errors import NoCompatibleInter

class InterObject(object):
    """This class represents operations taking place between two objects.

    Its instances have methods like join or copy_content or fetch, and contain
    references to the source and target objects these operations can be
    carried out between.

    Often we will provide convenience methods on the objects which carry out
    operations with another of similar type - they will always forward to
    a subclass of InterObject - i.e.
    InterVersionedFile.get(other).method_name(parameters).

    If the source and target objects implement the locking protocol -
    lock_read, lock_write, unlock, then the InterObject's lock_read,
    lock_write and unlock methods may be used (optionally in conjunction with
    the needs_read_lock and needs_write_lock decorators.)

    When looking for an inter, the most recently registered types are tested
    first.  So typically the most generic and slowest InterObjects should be
    registered first.
    """

    def __init__(self, source, target):
        if False:
            return 10
        "Construct a default InterObject instance. Please use 'get'.\n\n        Only subclasses of InterObject should call\n        InterObject.__init__ - clients should call InterFOO.get where FOO\n        is the base type of the objects they are interacting between. I.e.\n        InterVersionedFile or InterRepository.\n        get() is a convenience class method which will create an optimised\n        InterFOO if possible.\n        "
        self.source = source
        self.target = target

    def _double_lock(self, lock_source, lock_target):
        if False:
            print('Hello World!')
        'Take out two locks, rolling back the first if the second throws.'
        lock_source()
        try:
            lock_target()
        except Exception:
            self.source.unlock()
            raise

    @classmethod
    def get(klass, source, target):
        if False:
            while True:
                i = 10
        "Retrieve a Inter worker object for these objects.\n\n        :param source: the object to be the 'source' member of\n                       the InterObject instance.\n        :param target: the object to be the 'target' member of\n                       the InterObject instance.\n\n        If an optimised worker exists it will be used otherwise\n        a default Inter worker instance will be created.\n        "
        for provider in reversed(klass._optimisers):
            if provider.is_compatible(source, target):
                return provider(source, target)
        raise NoCompatibleInter(source, target)

    def lock_read(self):
        if False:
            return 10
        'Take out a logical read lock.\n\n        This will lock the source branch and the target branch. The source gets\n        a read lock and the target a read lock.\n        '
        self._double_lock(self.source.lock_read, self.target.lock_read)

    def lock_write(self):
        if False:
            print('Hello World!')
        'Take out a logical write lock.\n\n        This will lock the source branch and the target branch. The source gets\n        a read lock and the target a write lock.\n        '
        self._double_lock(self.source.lock_read, self.target.lock_write)

    @classmethod
    def register_optimiser(klass, optimiser):
        if False:
            while True:
                i = 10
        'Register an InterObject optimiser.'
        klass._optimisers.append(optimiser)

    def unlock(self):
        if False:
            return 10
        'Release the locks on source and target.'
        try:
            self.target.unlock()
        finally:
            self.source.unlock()

    @classmethod
    def unregister_optimiser(klass, optimiser):
        if False:
            print('Hello World!')
        'Unregister an InterObject optimiser.'
        klass._optimisers.remove(optimiser)