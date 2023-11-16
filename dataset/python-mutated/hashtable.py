class HashTable(object):
    """
    HashMap Data Type
    HashMap() Create a new, empty map. It returns an empty map collection.
    put(key, val) Add a new key-value pair to the map. If the key is already in the map then replace
                    the old value with the new value.
    get(key) Given a key, return the value stored in the map or None otherwise.
    del_(key) or del map[key] Delete the key-value pair from the map using a statement of the form del map[key].
    len() Return the number of key-value pairs stored in the map.
    in Return True for a statement of the form key in map, if the given key is in the map, False otherwise.
    """
    _empty = object()
    _deleted = object()

    def __init__(self, size=11):
        if False:
            return 10
        self.size = size
        self._len = 0
        self._keys = [self._empty] * size
        self._values = [self._empty] * size

    def put(self, key, value):
        if False:
            return 10
        initial_hash = hash_ = self.hash(key)
        while True:
            if self._keys[hash_] is self._empty or self._keys[hash_] is self._deleted:
                self._keys[hash_] = key
                self._values[hash_] = value
                self._len += 1
                return
            elif self._keys[hash_] == key:
                self._keys[hash_] = key
                self._values[hash_] = value
                return
            hash_ = self._rehash(hash_)
            if initial_hash == hash_:
                raise ValueError('Table is full')

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        initial_hash = hash_ = self.hash(key)
        while True:
            if self._keys[hash_] is self._empty:
                return None
            elif self._keys[hash_] == key:
                return self._values[hash_]
            hash_ = self._rehash(hash_)
            if initial_hash == hash_:
                return None

    def del_(self, key):
        if False:
            for i in range(10):
                print('nop')
        initial_hash = hash_ = self.hash(key)
        while True:
            if self._keys[hash_] is self._empty:
                return None
            elif self._keys[hash_] == key:
                self._keys[hash_] = self._deleted
                self._values[hash_] = self._deleted
                self._len -= 1
                return
            hash_ = self._rehash(hash_)
            if initial_hash == hash_:
                return None

    def hash(self, key):
        if False:
            return 10
        return key % self.size

    def _rehash(self, old_hash):
        if False:
            i = 10
            return i + 15
        '\n        linear probing\n        '
        return (old_hash + 1) % self.size

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.get(key)

    def __delitem__(self, key):
        if False:
            return 10
        return self.del_(key)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        self.put(key, value)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._len

class ResizableHashTable(HashTable):
    MIN_SIZE = 8

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(self.MIN_SIZE)

    def put(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        rv = super().put(key, value)
        if len(self) >= self.size * 2 / 3:
            self.__resize()

    def __resize(self):
        if False:
            for i in range(10):
                print('nop')
        (keys, values) = (self._keys, self._values)
        self.size *= 2
        self._len = 0
        self._keys = [self._empty] * self.size
        self._values = [self._empty] * self.size
        for (key, value) in zip(keys, values):
            if key is not self._empty and key is not self._deleted:
                self.put(key, value)