import threading
from func import synchronized
__all__ = ['LRU']

class LRUNode(object):
    __slots__ = ['prev', 'next', 'me']

    def __init__(self, prev, me):
        if False:
            while True:
                i = 10
        self.prev = prev
        self.me = me
        self.next = None

class LRU(object):
    """
    Implementation of a length-limited O(1) LRU queue.
    Built for and used by PyPE:
    http://pype.sourceforge.net
    Copyright 2003 Josiah Carlson.
    """

    def __init__(self, count, pairs=[]):
        if False:
            while True:
                i = 10
        self._lock = threading.RLock()
        self.count = max(count, 1)
        self.d = {}
        self.first = None
        self.last = None
        for (key, value) in pairs:
            self[key] = value

    @synchronized()
    def __contains__(self, obj):
        if False:
            print('Hello World!')
        return obj in self.d

    def get(self, obj, val=None):
        if False:
            print('Hello World!')
        try:
            return self[obj]
        except KeyError:
            return val

    @synchronized()
    def __getitem__(self, obj):
        if False:
            return 10
        a = self.d[obj].me
        self[a[0]] = a[1]
        return a[1]

    @synchronized()
    def __setitem__(self, obj, val):
        if False:
            print('Hello World!')
        if obj in self.d:
            del self[obj]
        nobj = LRUNode(self.last, (obj, val))
        if self.first is None:
            self.first = nobj
        if self.last:
            self.last.next = nobj
        self.last = nobj
        self.d[obj] = nobj
        if len(self.d) > self.count:
            if self.first == self.last:
                self.first = None
                self.last = None
                return
            a = self.first
            a.next.prev = None
            self.first = a.next
            a.next = None
            del self.d[a.me[0]]
            del a

    @synchronized()
    def __delitem__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        nobj = self.d[obj]
        if nobj.prev:
            nobj.prev.next = nobj.next
        else:
            self.first = nobj.next
        if nobj.next:
            nobj.next.prev = nobj.prev
        else:
            self.last = nobj.prev
        del self.d[obj]

    @synchronized()
    def __iter__(self):
        if False:
            i = 10
            return i + 15
        cur = self.first
        while cur is not None:
            cur2 = cur.next
            yield cur.me[1]
            cur = cur2

    @synchronized()
    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.d)

    @synchronized()
    def iteritems(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.first
        while cur is not None:
            cur2 = cur.next
            yield cur.me
            cur = cur2

    @synchronized()
    def iterkeys(self):
        if False:
            i = 10
            return i + 15
        return iter(self.d)

    @synchronized()
    def itervalues(self):
        if False:
            for i in range(10):
                print('nop')
        for (i, j) in self.iteritems():
            yield j

    @synchronized()
    def keys(self):
        if False:
            return 10
        return self.d.keys()

    @synchronized()
    def pop(self, key):
        if False:
            for i in range(10):
                print('nop')
        v = self[key]
        del self[key]
        return v

    @synchronized()
    def clear(self):
        if False:
            while True:
                i = 10
        self.d = {}
        self.first = None
        self.last = None