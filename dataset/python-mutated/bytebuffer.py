"""Implements ByteBuffer class for amortizing network transfer overhead."""
import io

class ByteBuffer(object):
    """Implements a byte buffer that allows callers to read data with minimal
    copying, and has a fast __len__ method. The buffer is parametrized by its
    chunk_size, which is the number of bytes that it will read in from the
    supplied reader or iterable when the buffer is being filled. As primary use
    case for this buffer is to amortize the overhead costs of transferring data
    over the network (rather than capping memory consumption), it leads to more
    predictable performance to always read the same amount of bytes each time
    the buffer is filled, hence the chunk_size parameter instead of some fixed
    capacity.

    The bytes are stored in a bytestring, and previously-read bytes are freed
    when the buffer is next filled (by slicing the bytestring into a smaller
    copy).

    Example
    -------

    Note that while this example works in both Python 2 and 3, the doctest only
    passes in Python 3 due to the bytestring literals in the expected values.

    >>> buf = ByteBuffer(chunk_size = 8)
    >>> message_bytes = iter([b'Hello, W', b'orld!'])
    >>> buf.fill(message_bytes)
    8
    >>> len(buf) # only chunk_size bytes are filled
    8
    >>> buf.peek()
    b'Hello, W'
    >>> len(buf) # peek() does not change read position
    8
    >>> buf.read(6)
    b'Hello,'
    >>> len(buf) # read() does change read position
    2
    >>> buf.fill(message_bytes)
    5
    >>> buf.read()
    b' World!'
    >>> len(buf)
    0
    """

    def __init__(self, chunk_size=io.DEFAULT_BUFFER_SIZE):
        if False:
            return 10
        'Create a ByteBuffer instance that reads chunk_size bytes when filled.\n        Note that the buffer has no maximum size.\n\n        Parameters\n        -----------\n        chunk_size: int, optional\n            The the number of bytes that will be read from the supplied reader\n            or iterable when filling the buffer.\n        '
        self._chunk_size = chunk_size
        self.empty()

    def __len__(self):
        if False:
            return 10
        'Return the number of unread bytes in the buffer as an int'
        return len(self._bytes) - self._pos

    def read(self, size=-1):
        if False:
            for i in range(10):
                print('nop')
        'Read bytes from the buffer and advance the read position. Returns\n        the bytes in a bytestring.\n\n        Parameters\n        ----------\n        size: int, optional\n            Maximum number of bytes to read. If negative or not supplied, read\n            all unread bytes in the buffer.\n\n        Returns\n        -------\n        bytes\n        '
        part = self.peek(size)
        self._pos += len(part)
        return part

    def peek(self, size=-1):
        if False:
            print('Hello World!')
        'Get bytes from the buffer without advancing the read position.\n        Returns the bytes in a bytestring.\n\n        Parameters\n        ----------\n        size: int, optional\n            Maximum number of bytes to return. If negative or not supplied,\n            return all unread bytes in the buffer.\n\n        Returns\n        -------\n        bytes\n        '
        if size < 0 or size > len(self):
            size = len(self)
        part = bytes(self._bytes[self._pos:self._pos + size])
        return part

    def empty(self):
        if False:
            i = 10
            return i + 15
        'Remove all bytes from the buffer'
        self._bytes = bytearray()
        self._pos = 0

    def fill(self, source, size=-1):
        if False:
            return 10
        "Fill the buffer with bytes from source until one of these\n        conditions is met:\n            * size bytes have been read from source (if size >= 0);\n            * chunk_size bytes have been read from source;\n            * no more bytes can be read from source;\n        Returns the number of new bytes added to the buffer.\n        Note: all previously-read bytes in the buffer are removed.\n\n        Parameters\n        ----------\n        source: a file-like object, or iterable/list that contains bytes\n            The source of bytes to fill the buffer with. If this argument has\n            the `read` attribute, it's assumed to be a file-like object and\n            `read` is called to get the bytes; otherwise it's assumed to be an\n            iterable or list that contains bytes, and a for loop is used to get\n            the bytes.\n        size: int, optional\n            The number of bytes to try to read from source. If not supplied,\n            negative, or larger than the buffer's chunk_size, then chunk_size\n            bytes are read. Note that if source is an iterable or list, then\n            it's possible that more than size bytes will be read if iterating\n            over source produces more than one byte at a time.\n\n        Returns\n        -------\n        int, the number of new bytes added to the buffer.\n        "
        size = size if size >= 0 else self._chunk_size
        size = min(size, self._chunk_size)
        if self._pos != 0:
            self._bytes = self._bytes[self._pos:]
            self._pos = 0
        if hasattr(source, 'read'):
            new_bytes = source.read(size)
        else:
            new_bytes = bytearray()
            for more_bytes in source:
                new_bytes += more_bytes
                if len(new_bytes) >= size:
                    break
        self._bytes += new_bytes
        return len(new_bytes)

    def readline(self, terminator):
        if False:
            while True:
                i = 10
        'Read a line from this buffer efficiently.\n\n        A line is a contiguous sequence of bytes that ends with either:\n\n        1. The ``terminator`` character\n        2. The end of the buffer itself\n\n        :param byte terminator: The line terminator character.\n        :rtype: bytes\n\n        '
        index = self._bytes.find(terminator, self._pos)
        if index == -1:
            size = len(self)
        else:
            size = index - self._pos + 1
        return self.read(size)