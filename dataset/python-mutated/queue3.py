from cython.cimports import cqueue
from cython import cast

@cython.cclass
class Queue:
    """A queue class for C integer values.

    >>> q = Queue()
    >>> q.append(5)
    >>> q.peek()
    5
    >>> q.pop()
    5
    """
    _c_queue = cython.declare(cython.pointer(cqueue.Queue))

    def __cinit__(self):
        if False:
            return 10
        self._c_queue = cqueue.queue_new()
        if self._c_queue is cython.NULL:
            raise MemoryError()

    def __dealloc__(self):
        if False:
            i = 10
            return i + 15
        if self._c_queue is not cython.NULL:
            cqueue.queue_free(self._c_queue)

    @cython.ccall
    def append(self, value: cython.int):
        if False:
            while True:
                i = 10
        if not cqueue.queue_push_tail(self._c_queue, cast(cython.p_void, cast(cython.Py_ssize_t, value))):
            raise MemoryError()

    @cython.ccall
    def extend(self, values):
        if False:
            for i in range(10):
                print('nop')
        for value in values:
            self.append(value)

    @cython.cfunc
    def extend_ints(self, values: cython.p_int, count: cython.size_t):
        if False:
            return 10
        value: cython.int
        for value in values[:count]:
            self.append(value)

    @cython.ccall
    @cython.exceptval(-1, check=True)
    def peek(self) -> cython.int:
        if False:
            print('Hello World!')
        value: cython.int = cast(cython.Py_ssize_t, cqueue.queue_peek_head(self._c_queue))
        if value == 0:
            if cqueue.queue_is_empty(self._c_queue):
                raise IndexError('Queue is empty')
        return value

    @cython.ccall
    @cython.exceptval(-1, check=True)
    def pop(self) -> cython.int:
        if False:
            while True:
                i = 10
        if cqueue.queue_is_empty(self._c_queue):
            raise IndexError('Queue is empty')
        return cast(cython.Py_ssize_t, cqueue.queue_pop_head(self._c_queue))

    def __bool__(self):
        if False:
            return 10
        return not cqueue.queue_is_empty(self._c_queue)