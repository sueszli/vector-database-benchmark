import heapq

class FileSharing(object):

    def __init__(self, m):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        '
        self.__users = []
        self.__lookup = set()
        self.__min_heap = []

    def join(self, ownedChunks):
        if False:
            while True:
                i = 10
        '\n        :type ownedChunks: List[int]\n        :rtype: int\n        '
        if self.__min_heap:
            userID = heapq.heappop(self.__min_heap)
        else:
            userID = len(self.__users) + 1
            self.__users.append(set())
        self.__users[userID - 1] = set(ownedChunks)
        self.__lookup.add(userID)
        return userID

    def leave(self, userID):
        if False:
            i = 10
            return i + 15
        '\n        :type userID: int\n        :rtype: None\n        '
        if userID not in self.__lookup:
            return
        self.__lookup.remove(userID)
        self.__users[userID - 1] = []
        heapq.heappush(self.__min_heap, userID)

    def request(self, userID, chunkID):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type userID: int\n        :type chunkID: int\n        :rtype: List[int]\n        '
        result = []
        for (u, chunks) in enumerate(self.__users, 1):
            if chunkID not in chunks:
                continue
            result.append(u)
        if not result:
            return
        self.__users[userID - 1].add(chunkID)
        return result
import collections
import heapq

class FileSharing2(object):

    def __init__(self, m):
        if False:
            return 10
        '\n        :type m: int\n        '
        self.__users = []
        self.__lookup = set()
        self.__chunks = collections.defaultdict(set)
        self.__min_heap = []

    def join(self, ownedChunks):
        if False:
            return 10
        '\n        :type ownedChunks: List[int]\n        :rtype: int\n        '
        if self.__min_heap:
            userID = heapq.heappop(self.__min_heap)
        else:
            userID = len(self.__users) + 1
            self.__users.append(set())
        self.__users[userID - 1] = set(ownedChunks)
        self.__lookup.add(userID)
        for c in ownedChunks:
            self.__chunks[c].add(userID)
        return userID

    def leave(self, userID):
        if False:
            while True:
                i = 10
        '\n        :type userID: int\n        :rtype: None\n        '
        if userID not in self.__lookup:
            return
        for c in self.__users[userID - 1]:
            self.__chunks[c].remove(userID)
        self.__lookup.remove(userID)
        self.__users[userID - 1] = []
        heapq.heappush(self.__min_heap, userID)

    def request(self, userID, chunkID):
        if False:
            print('Hello World!')
        '\n        :type userID: int\n        :type chunkID: int\n        :rtype: List[int]\n        '
        result = sorted(self.__chunks[chunkID])
        if not result:
            return
        self.__users[userID - 1].add(chunkID)
        self.__chunks[chunkID].add(userID)
        return result