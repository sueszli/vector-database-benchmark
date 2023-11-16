import array
import time

class PeerHashfield(object):
    __slots__ = ('storage', 'time_changed', 'append', 'remove', 'tobytes', 'frombytes', '__len__', '__iter__')

    def __init__(self):
        if False:
            return 10
        self.storage = self.createStorage()
        self.time_changed = time.time()

    def createStorage(self):
        if False:
            for i in range(10):
                print('nop')
        storage = array.array('H')
        self.append = storage.append
        self.remove = storage.remove
        self.tobytes = storage.tobytes
        self.frombytes = storage.frombytes
        self.__len__ = storage.__len__
        self.__iter__ = storage.__iter__
        return storage

    def appendHash(self, hash):
        if False:
            print('Hello World!')
        hash_id = int(hash[0:4], 16)
        if hash_id not in self.storage:
            self.storage.append(hash_id)
            self.time_changed = time.time()
            return True
        else:
            return False

    def appendHashId(self, hash_id):
        if False:
            for i in range(10):
                print('nop')
        if hash_id not in self.storage:
            self.storage.append(hash_id)
            self.time_changed = time.time()
            return True
        else:
            return False

    def removeHash(self, hash):
        if False:
            i = 10
            return i + 15
        hash_id = int(hash[0:4], 16)
        if hash_id in self.storage:
            self.storage.remove(hash_id)
            self.time_changed = time.time()
            return True
        else:
            return False

    def removeHashId(self, hash_id):
        if False:
            while True:
                i = 10
        if hash_id in self.storage:
            self.storage.remove(hash_id)
            self.time_changed = time.time()
            return True
        else:
            return False

    def getHashId(self, hash):
        if False:
            for i in range(10):
                print('nop')
        return int(hash[0:4], 16)

    def hasHash(self, hash):
        if False:
            i = 10
            return i + 15
        return int(hash[0:4], 16) in self.storage

    def replaceFromBytes(self, hashfield_raw):
        if False:
            for i in range(10):
                print('nop')
        self.storage = self.createStorage()
        self.storage.frombytes(hashfield_raw)
        self.time_changed = time.time()
if __name__ == '__main__':
    field = PeerHashfield()
    s = time.time()
    for i in range(10000):
        field.appendHashId(i)
    print(time.time() - s)
    s = time.time()
    for i in range(10000):
        field.hasHash('AABB')
    print(time.time() - s)