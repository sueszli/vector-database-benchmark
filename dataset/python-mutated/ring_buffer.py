"""Implementation of a ring buffer using bytearray."""

class RingBuffer:
    """Basic ring buffer using a bytearray.

    Not threadsafe.
    """

    def __init__(self, maxlen: int) -> None:
        if False:
            return 10
        'Initialize empty buffer.'
        self._buffer = bytearray(maxlen)
        self._pos = 0
        self._length = 0
        self._maxlen = maxlen

    @property
    def maxlen(self) -> int:
        if False:
            while True:
                i = 10
        'Return the maximum size of the buffer.'
        return self._maxlen

    @property
    def pos(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the current put position.'
        return self._pos

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the length of data stored in the buffer.'
        return self._length

    def put(self, data: bytes) -> None:
        if False:
            print('Hello World!')
        'Put a chunk of data into the buffer, possibly wrapping around.'
        data_len = len(data)
        new_pos = self._pos + data_len
        if new_pos >= self._maxlen:
            num_bytes_1 = self._maxlen - self._pos
            num_bytes_2 = new_pos - self._maxlen
            self._buffer[self._pos:self._maxlen] = data[:num_bytes_1]
            self._buffer[:num_bytes_2] = data[num_bytes_1:]
            new_pos = new_pos - self._maxlen
        else:
            self._buffer[self._pos:self._pos + data_len] = data
        self._pos = new_pos
        self._length = min(self._maxlen, self._length + data_len)

    def getvalue(self) -> bytes:
        if False:
            while True:
                i = 10
        'Get bytes written to the buffer.'
        if self._pos + self._length <= self._maxlen:
            return bytes(self._buffer[:self._length])
        return bytes(self._buffer[self._pos:] + self._buffer[:self._pos])