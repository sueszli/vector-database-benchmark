class Barrier:
    """A distributed barrier.

    Examples:

      >>> from dramatiq.rate_limits import Barrier
      >>> from dramatiq.rate_limits.backends import RedisBackend

      >>> backend = RedisBackend()
      >>> barrier = Barrier(backend, "some-barrier", ttl=30_000)

      >>> created = barrier.create(parties=3)
      >>> barrier.wait(block=False)
      False
      >>> barrier.wait(block=False)
      False
      >>> barrier.wait(block=False)
      True

    Parameters:
      backend(BarrierBackend): The barrier backend to use.
      key(str): The key for the barrier.
      ttl(int): The TTL for the barrier key, in milliseconds.
    """

    def __init__(self, backend, key, *, ttl=900000):
        if False:
            i = 10
            return i + 15
        self.backend = backend
        self.key = key
        self.key_events = key + '@events'
        self.ttl = ttl

    def create(self, parties):
        if False:
            i = 10
            return i + 15
        'Create the barrier for the given number of parties.\n\n        Parameters:\n          parties(int): The number of parties to wait for.\n\n        Returns:\n          bool: Whether or not the new barrier was successfully created.\n        '
        assert parties > 0, 'parties must be a positive integer.'
        return self.backend.add(self.key, parties, self.ttl)

    def wait(self, *, block=True, timeout=None):
        if False:
            i = 10
            return i + 15
        'Signal that a party has reached the barrier.\n\n        Warning:\n          Barrier blocking is currently only supported by the stub and\n          Redis backends.\n\n        Warning:\n          Re-using keys between blocking calls may lead to undefined\n          behaviour.  Make sure your barrier keys are always unique\n          (use a UUID).\n\n        Parameters:\n          block(bool): Whether or not to block while waiting for the\n            other parties.\n          timeout(int): The maximum number of milliseconds to wait for\n            the barrier to be cleared.\n\n        Returns:\n          bool: Whether or not the barrier has been reached by all parties.\n        '
        cleared = not self.backend.decr(self.key, 1, 1, self.ttl)
        if cleared:
            self.backend.wait_notify(self.key_events, self.ttl)
            return True
        if block:
            return self.backend.wait(self.key_events, timeout)
        return False