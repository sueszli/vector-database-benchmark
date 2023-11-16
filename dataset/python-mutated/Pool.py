"""
Pool is a collection of python objects that you can checkin and
checkout. This is useful for a cache of objects that are expensive to load
and can be reused over and over, like splashes on cannonballs, or
bulletholes on walls. The pool is unsorted. Items do not have to be unique
or be the same type.

Internally the pool is implemented with 2 lists, free items and used items.

Example:

    .. code-block:: python

        p = Pool([1, 2, 3, 4, 5])
        x = p.checkout()
        p.checkin(x)

"""
__all__ = ['Pool']
from direct.directnotify import DirectNotifyGlobal

class Pool:
    notify = DirectNotifyGlobal.directNotify.newCategory('Pool')

    def __init__(self, free=None):
        if False:
            i = 10
            return i + 15
        if free:
            self.__free = free
        else:
            self.__free = []
        self.__used = []

    def add(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an item to the free list.\n        '
        self.__free.append(item)

    def remove(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Remove an item. Error is flagged if the item is not in the pool.\n        '
        if item in self.__free:
            self.__free.remove(item)
        elif item in self.__used:
            self.__used.remove(item)
        else:
            self.notify.error('item not in pool')

    def checkout(self):
        if False:
            print('Hello World!')
        '\n        Get an arbitrary item from the pool.\n        '
        if not self.__free:
            self.notify.error('no items are free')
        item = self.__free.pop()
        self.__used.append(item)
        return item

    def checkin(self, item):
        if False:
            while True:
                i = 10
        '\n        Put back a checked out item.\n        Error if the item is not checked out.\n        '
        if item not in self.__used:
            self.notify.error('item is not checked out')
        self.__used.remove(item)
        self.__free.append(item)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Resets the pool so all items are free.\n        '
        self.__free.extend(self.__used)
        self.__used = []

    def hasFree(self):
        if False:
            while True:
                i = 10
        '\n        Returns true if there is at least one free item.\n        '
        return len(self.__free) != 0

    def isFree(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns true if this item is free for check out.\n        '
        return item in self.__free

    def isUsed(self, item):
        if False:
            print('Hello World!')
        '\n        Returns true if this item has already been checked out.\n        '
        return item in self.__used

    def getNumItems(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of free items and the number of used items.\n        '
        return (len(self.__free), len(self.__used))

    def cleanup(self, cleanupFunc=None):
        if False:
            return 10
        '\n        Completely cleanup the pool and all of its objects.\n        cleanupFunc will be called on every free and used item.\n        '
        if cleanupFunc:
            allItems = self.__free + self.__used
            for item in allItems:
                cleanupFunc(item)
        del self.__free
        del self.__used

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'free = %s\nused = %s' % (self.__free, self.__used)