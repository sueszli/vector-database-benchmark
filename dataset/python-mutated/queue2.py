from cython.cimports import cqueue

@cython.cclass
class Queue:
    _c_queue = cython.declare(cython.pointer(cqueue.Queue))

    def __cinit__(self):
        if False:
            i = 10
            return i + 15
        self._c_queue = cqueue.queue_new()
        if self._c_queue is cython.NULL:
            raise MemoryError()