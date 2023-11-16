"""
Provides an interface for a fast FIFO buffer.

The interface implements only 'read()', 'write()' and 'len()'.  The
implementation below is a modified version of the code originally written by
Ben Timby: http://ben.timby.com/?p=139
"""
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
MAX_BUFFER = 1024 ** 2 * 4

class Buffer(object):
    """
    Implements a fast FIFO buffer.

    Internally, the buffer consists of a list of StringIO objects.  New
    StringIO objects are added and delete as data is written to and read from
    the FIFO buffer.
    """

    def __init__(self, max_size=MAX_BUFFER):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialise a Buffer object.\n        '
        self.buffers = []
        self.max_size = max_size
        self.read_pos = 0
        self.write_pos = 0

    def write(self, data):
        if False:
            i = 10
            return i + 15
        "\n        Write `data' to the FIFO buffer.\n\n        If necessary, a new internal buffer is created.\n        "
        if not self.buffers:
            self.buffers.append(StringIO())
            self.write_pos = 0
        lastBuf = self.buffers[-1]
        lastBuf.seek(self.write_pos)
        lastBuf.write(data)
        if lastBuf.tell() >= self.max_size:
            lastBuf = StringIO()
            self.buffers.append(lastBuf)
        self.write_pos = lastBuf.tell()

    def read(self, length=-1):
        if False:
            i = 10
            return i + 15
        "\n        Read `length' elements of the FIFO buffer.\n\n        Drained data is automatically deleted.\n        "
        read_buf = StringIO()
        remaining = length
        while True:
            if not self.buffers:
                break
            firstBuf = self.buffers[0]
            firstBuf.seek(self.read_pos)
            read_buf.write(firstBuf.read(remaining))
            self.read_pos = firstBuf.tell()
            if length == -1:
                del self.buffers[0]
                self.read_pos = 0
            else:
                remaining = length - read_buf.tell()
                if remaining > 0:
                    del self.buffers[0]
                    self.read_pos = 0
                else:
                    break
        return read_buf.getvalue()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the length of the Buffer object.\n        '
        length = 0
        for buf in self.buffers:
            buf.seek(0, 2)
            if buf == self.buffers[0]:
                length += buf.tell() - self.read_pos
            else:
                length += buf.tell()
        return length