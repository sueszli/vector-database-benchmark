from __future__ import absolute_import, division, print_function, unicode_literals
'\n    sockjs.tornado.sessioncontainer\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    Simple heapq-based session implementation with sliding expiration window\n    support.\n'
from heapq import heappush, heappop
from time import time
from hashlib import md5
from random import random

def _random_key():
    if False:
        print('Hello World!')
    'Return random session key'
    i = md5()
    i.update('%s%s' % (random(), time()))
    return i.hexdigest()

class SessionMixin(object):
    """Represents one session object stored in the session container.
    Derive from this object to store additional data.
    """

    def __init__(self, session_id=None, expiry=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n        ``session_id``\n            Optional session id. If not provided, will generate\n            new session id.\n        ``expiry``\n            Expiration time. If not provided, will never expire.\n        '
        self.session_id = session_id or _random_key()
        self.promoted = None
        self.expiry = expiry
        if self.expiry is not None:
            self.expiry_date = time() + self.expiry

    def is_alive(self):
        if False:
            while True:
                i = 10
        'Check if session is still alive'
        return self.expiry_date > time()

    def promote(self):
        if False:
            return 10
        "Mark object as alive, so it won't be collected during next\n        run of the garbage collector.\n        "
        if self.expiry is not None:
            self.promoted = time() + self.expiry

    def on_delete(self, forced):
        if False:
            i = 10
            return i + 15
        'Triggered when object was expired or deleted.'
        pass

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.expiry_date < other.expiry_date
    __cmp__ = __lt__

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%f %s %d' % (getattr(self, 'expiry_date', -1), self.session_id, self.promoted or 0)

class SessionContainer(object):
    """Session container object.

    If we will implement sessions with Tornado timeouts, for polling transports
    it will be nightmare - if load will be high, number of discarded timeouts
    will be huge and will be huge performance hit, as Tornado will have to
    clean them up all the time.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._items = {}
        self._queue = []

    def add(self, session):
        if False:
            print('Hello World!')
        'Add session to the container.\n\n        `session`\n            Session object\n        '
        self._items[session.session_id] = session
        if session.expiry is not None:
            heappush(self._queue, session)

    def get(self, session_id):
        if False:
            while True:
                i = 10
        'Return session object or None if it is not available\n\n        `session_id`\n            Session identifier\n        '
        return self._items.get(session_id, None)

    def remove(self, session_id):
        if False:
            for i in range(10):
                print('nop')
        'Remove session object from the container\n\n        `session_id`\n            Session identifier\n        '
        session = self._items.get(session_id, None)
        if session is not None:
            session.promoted = -1
            session.on_delete(True)
            del self._items[session_id]
            return True
        return False

    def expire(self, current_time=None):
        if False:
            i = 10
            return i + 15
        'Expire any old entries\n\n        `current_time`\n            Optional time to be used to clean up queue (can be used in unit tests)\n        '
        if not self._queue:
            return
        if current_time is None:
            current_time = time()
        while self._queue:
            top = self._queue[0]
            if top.promoted is None and top.expiry_date > current_time:
                break
            top = heappop(self._queue)
            need_reschedule = top.promoted is not None and top.promoted > current_time
            if not need_reschedule:
                top.promoted = None
                top.on_delete(False)
                need_reschedule = top.promoted is not None and top.promoted > current_time
            if need_reschedule:
                top.expiry_date = top.promoted
                top.promoted = None
                heappush(self._queue, top)
            else:
                del self._items[top.session_id]