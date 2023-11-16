"""
*What is this pattern about?
This pattern is used when creating an object is costly (and they are
created frequently) but only a few are used at a time. With a Pool we
can manage those instances we have as of now by caching them. Now it
is possible to skip the costly creation of an object if one is
available in the pool.
A pool allows to 'check out' an inactive object and then to return it.
If none are available the pool creates one to provide without wait.

*What does this example do?
In this example queue.Queue is used to create the pool (wrapped in a
custom ObjectPool object to use with the with statement), and it is
populated with strings.
As we can see, the first string object put in "yam" is USED by the
with statement. But because it is released back into the pool
afterwards it is reused by the explicit call to sample_queue.get().
Same thing happens with "sam", when the ObjectPool created inside the
function is deleted (by the GC) and the object is returned.

*Where is the pattern used practically?

*References:
http://stackoverflow.com/questions/1514120/python-implementation-of-the-object-pool-design-pattern
https://sourcemaking.com/design_patterns/object_pool

*TL;DR
Stores a set of initialized objects kept ready to use.
"""

class ObjectPool:

    def __init__(self, queue, auto_get=False):
        if False:
            while True:
                i = 10
        self._queue = queue
        self.item = self._queue.get() if auto_get else None

    def __enter__(self):
        if False:
            while True:
                i = 10
        if self.item is None:
            self.item = self._queue.get()
        return self.item

    def __exit__(self, Type, value, traceback):
        if False:
            return 10
        if self.item is not None:
            self._queue.put(self.item)
            self.item = None

    def __del__(self):
        if False:
            print('Hello World!')
        if self.item is not None:
            self._queue.put(self.item)
            self.item = None

def main():
    if False:
        return 10
    "\n    >>> import queue\n\n    >>> def test_object(queue):\n    ...    pool = ObjectPool(queue, True)\n    ...    print('Inside func: {}'.format(pool.item))\n\n    >>> sample_queue = queue.Queue()\n\n    >>> sample_queue.put('yam')\n    >>> with ObjectPool(sample_queue) as obj:\n    ...    print('Inside with: {}'.format(obj))\n    Inside with: yam\n\n    >>> print('Outside with: {}'.format(sample_queue.get()))\n    Outside with: yam\n\n    >>> sample_queue.put('sam')\n    >>> test_object(sample_queue)\n    Inside func: sam\n\n    >>> print('Outside func: {}'.format(sample_queue.get()))\n    Outside func: sam\n\n    if not sample_queue.empty():\n        print(sample_queue.get())\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()