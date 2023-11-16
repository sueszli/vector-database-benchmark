import io

class ContainerIO:
    """
    A file object that provides read access to a part of an existing
    file (for example a TAR file).
    """

    def __init__(self, file, offset, length):
        if False:
            return 10
        '\n        Create file object.\n\n        :param file: Existing file.\n        :param offset: Start of region, in bytes.\n        :param length: Size of region, in bytes.\n        '
        self.fh = file
        self.pos = 0
        self.offset = offset
        self.length = length
        self.fh.seek(offset)

    def isatty(self):
        if False:
            while True:
                i = 10
        return False

    def seek(self, offset, mode=io.SEEK_SET):
        if False:
            return 10
        '\n        Move file pointer.\n\n        :param offset: Offset in bytes.\n        :param mode: Starting position. Use 0 for beginning of region, 1\n           for current offset, and 2 for end of region.  You cannot move\n           the pointer outside the defined region.\n        '
        if mode == 1:
            self.pos = self.pos + offset
        elif mode == 2:
            self.pos = self.length + offset
        else:
            self.pos = offset
        self.pos = max(0, min(self.pos, self.length))
        self.fh.seek(self.offset + self.pos)

    def tell(self):
        if False:
            print('Hello World!')
        '\n        Get current file pointer.\n\n        :returns: Offset from start of region, in bytes.\n        '
        return self.pos

    def read(self, n=0):
        if False:
            i = 10
            return i + 15
        '\n        Read data.\n\n        :param n: Number of bytes to read. If omitted or zero,\n            read until end of region.\n        :returns: An 8-bit string.\n        '
        if n:
            n = min(n, self.length - self.pos)
        else:
            n = self.length - self.pos
        if not n:
            return b'' if 'b' in self.fh.mode else ''
        self.pos = self.pos + n
        return self.fh.read(n)

    def readline(self):
        if False:
            i = 10
            return i + 15
        '\n        Read a line of text.\n\n        :returns: An 8-bit string.\n        '
        s = b'' if 'b' in self.fh.mode else ''
        newline_character = b'\n' if 'b' in self.fh.mode else '\n'
        while True:
            c = self.read(1)
            if not c:
                break
            s = s + c
            if c == newline_character:
                break
        return s

    def readlines(self):
        if False:
            while True:
                i = 10
        '\n        Read multiple lines of text.\n\n        :returns: A list of 8-bit strings.\n        '
        lines = []
        while True:
            s = self.readline()
            if not s:
                break
            lines.append(s)
        return lines