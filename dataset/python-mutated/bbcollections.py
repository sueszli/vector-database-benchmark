from collections import defaultdict
assert defaultdict

class KeyedSets:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.d = {}

    def add(self, key, value):
        if False:
            return 10
        if key not in self.d:
            self.d[key] = set()
        self.d[key].add(value)

    def discard(self, key, value):
        if False:
            while True:
                i = 10
        if key in self.d:
            self.d[key].discard(value)
            if not self.d[key]:
                del self.d[key]

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return key in self.d

    def __getitem__(self, key):
        if False:
            return 10
        return self.d.get(key, set())

    def pop(self, key):
        if False:
            while True:
                i = 10
        if key in self.d:
            return self.d.pop(key)
        return set()