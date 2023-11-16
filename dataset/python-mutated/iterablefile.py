from __future__ import absolute_import

class IterableFileBase(object):
    """Create a file-like object from any iterable"""

    def __init__(self, iterable):
        if False:
            print('Hello World!')
        object.__init__(self)
        self._iter = iterable.__iter__()
        self._buffer = ''
        self.done = False

    def read_n(self, length):
        if False:
            return 10
        "\n        >>> IterableFileBase(['This ', 'is ', 'a ', 'test.']).read_n(8)\n        'This is '\n        "

        def test_length(result):
            if False:
                for i in range(10):
                    print('nop')
            if len(result) >= length:
                return length
            else:
                return None
        return self._read(test_length)

    def read_to(self, sequence, length=None):
        if False:
            while True:
                i = 10
        "\n        >>> f = IterableFileBase(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.read_to('\\n')\n        'Th\\n'\n        >>> f.read_to('\\n')\n        'is is \\n'\n        "

        def test_contents(result):
            if False:
                print('Hello World!')
            if length is not None:
                if len(result) >= length:
                    return length
            try:
                return result.index(sequence) + len(sequence)
            except ValueError:
                return None
        return self._read(test_contents)

    def _read(self, result_length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read data until result satisfies the condition result_length.\n        result_length is a callable that returns None until the condition\n        is satisfied, and returns the length of the result to use when\n        the condition is satisfied.  (i.e. it returns the length of the\n        subset of the first condition match.)\n        '
        result = self._buffer
        while result_length(result) is None:
            try:
                result += self._iter.next()
            except StopIteration:
                self.done = True
                self._buffer = ''
                return result
        output_length = result_length(result)
        self._buffer = result[output_length:]
        return result[:output_length]

    def read_all(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> IterableFileBase(['This ', 'is ', 'a ', 'test.']).read_all()\n        'This is a test.'\n        "

        def no_stop(result):
            if False:
                i = 10
                return i + 15
            return None
        return self._read(no_stop)

    def push_back(self, contents):
        if False:
            return 10
        '\n        >>> f = IterableFileBase([\'Th\\nis \', \'is \\n\', \'a \', \'te\\nst.\'])\n        >>> f.read_to(\'\\n\')\n        \'Th\\n\'\n        >>> f.push_back("Sh")\n        >>> f.read_all()\n        \'Shis is \\na te\\nst.\'\n        '
        self._buffer = contents + self._buffer

class IterableFile(object):
    """This class supplies all File methods that can be implemented cheaply."""

    def __init__(self, iterable):
        if False:
            return 10
        object.__init__(self)
        self._file_base = IterableFileBase(iterable)
        self._iter = self._make_iterator()
        self._closed = False
        self.softspace = 0

    def _make_iterator(self):
        if False:
            print('Hello World!')
        while not self._file_base.done:
            self._check_closed()
            result = self._file_base.read_to('\n')
            if result != '':
                yield result

    def _check_closed(self):
        if False:
            print('Hello World!')
        if self.closed:
            raise ValueError('File is closed.')

    def close(self):
        if False:
            while True:
                i = 10
        "\n        >>> f = IterableFile(['This ', 'is ', 'a ', 'test.'])\n        >>> f.closed\n        False\n        >>> f.close()\n        >>> f.closed\n        True\n        "
        self._file_base.done = True
        self._closed = True
    closed = property(lambda x: x._closed)

    def flush(self):
        if False:
            return 10
        'No-op for standard compliance.\n        >>> f = IterableFile([])\n        >>> f.close()\n        >>> f.flush()\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        '
        self._check_closed()

    def next(self):
        if False:
            print('Hello World!')
        "Implementation of the iterator protocol's next()\n\n        >>> f = IterableFile(['This \\n', 'is ', 'a ', 'test.'])\n        >>> f.next()\n        'This \\n'\n        >>> f.close()\n        >>> f.next()\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        >>> f = IterableFile(['This \\n', 'is ', 'a ', 'test.\\n'])\n        >>> f.next()\n        'This \\n'\n        >>> f.next()\n        'is a test.\\n'\n        >>> f.next()\n        Traceback (most recent call last):\n        StopIteration\n        "
        self._check_closed()
        return self._iter.next()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> list(IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.']))\n        ['Th\\n', 'is is \\n', 'a te\\n', 'st.']\n        >>> f = IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.close()\n        >>> list(f)\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        "
        return self

    def read(self, length=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> IterableFile(['This ', 'is ', 'a ', 'test.']).read()\n        'This is a test.'\n        >>> f = IterableFile(['This ', 'is ', 'a ', 'test.'])\n        >>> f.read(10)\n        'This is a '\n        >>> f = IterableFile(['This ', 'is ', 'a ', 'test.'])\n        >>> f.close()\n        >>> f.read(10)\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        "
        self._check_closed()
        if length is None:
            return self._file_base.read_all()
        else:
            return self._file_base.read_n(length)

    def read_to(self, sequence, size=None):
        if False:
            i = 10
            return i + 15
        "\n        Read characters until a sequence is found, with optional max size.\n        The specified sequence, if found, will be included in the result\n\n        >>> f = IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.read_to('i')\n        'Th\\ni'\n        >>> f.read_to('i')\n        's i'\n        >>> f.close()\n        >>> f.read_to('i')\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        "
        self._check_closed()
        return self._file_base.read_to(sequence, size)

    def readline(self, size=None):
        if False:
            while True:
                i = 10
        "\n        >>> f = IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.readline()\n        'Th\\n'\n        >>> f.readline(4)\n        'is i'\n        >>> f.close()\n        >>> f.readline()\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        "
        return self.read_to('\n', size)

    def readlines(self, sizehint=None):
        if False:
            i = 10
            return i + 15
        "\n        >>> f = IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.readlines()\n        ['Th\\n', 'is is \\n', 'a te\\n', 'st.']\n        >>> f = IterableFile(['Th\\nis ', 'is \\n', 'a ', 'te\\nst.'])\n        >>> f.close()\n        >>> f.readlines()\n        Traceback (most recent call last):\n        ValueError: File is closed.\n        "
        lines = []
        while True:
            line = self.readline()
            if line == '':
                return lines
            if sizehint is None:
                lines.append(line)
            elif len(line) < sizehint:
                lines.append(line)
                sizehint -= len(line)
            else:
                self._file_base.push_back(line)
                return lines
if __name__ == '__main__':
    import doctest
    doctest.testmod()