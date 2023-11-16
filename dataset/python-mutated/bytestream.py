class ByteStream(object):
    """Basic interface that reads and writes bytes"""

    def read(self, count, timeout=None):
        if False:
            i = 10
            return i + 15
        '\n        Reads exactly count bytes from the stream. This call is blocking until count bytes\n        are read or an error happens\n\n        This call returns a byte array or EOFError if there was a problem\n        reading.\n\n        Parameters\n        ----------\n        count : int\n            Exact number of characters to read\n\n        Returns\n        -------\n        bytes\n            Content read from the stream\n\n        Raises\n        ------\n        EOFError\n            Any issue with reading will be raised as a EOFError\n        '
        raise NotImplementedError

    def write(self, data):
        if False:
            print('Hello World!')
        '\n        Writes all the data to the stream\n\n        This call is blocking until all data is written. EOFError will be\n        raised if there is a problem writing to the stream\n\n        Parameters\n        ----------\n        data : bytes\n            Data to write out\n\n        Raises\n        ------\n        EOFError\n            Any issue with writing will be raised as a EOFError\n        '
        raise NotImplementedError

    def close(self):
        if False:
            print('Hello World!')
        '\n        Closes the stream releasing all system resources\n\n        Once closed, the stream cannot be re-opened or re-used. If a\n        stream is already closed, this operation will have no effect\n        '
        raise NotImplementedError()

    @property
    def is_closed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the stream is closed or False otherwise\n\n        Returns\n        -------\n        bool\n            True if closed or False otherwise\n        '
        raise NotImplementedError()

    def fileno(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *exc_info):
        if False:
            for i in range(10):
                print('nop')
        self.close()