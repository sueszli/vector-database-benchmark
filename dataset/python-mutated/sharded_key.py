class ShardedKey(object):
    """
  A sharded key consisting of a user key and an opaque shard id represented by
  bytes.

  Attributes:
    key: The user key.
    shard_id: An opaque byte string that uniquely represents a shard of the key.
  """

    def __init__(self, key, shard_id):
        if False:
            for i in range(10):
                print('nop')
        assert shard_id is not None
        self._key = key
        self._shard_id = shard_id

    @property
    def key(self):
        if False:
            for i in range(10):
                print('nop')
        return self._key

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '(%s, %s)' % (repr(self.key), self._shard_id)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return type(self) == type(other) and self.key == other.key and (self._shard_id == other._shard_id)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.key, self._shard_id))

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (ShardedKey, (self.key, self._shard_id))