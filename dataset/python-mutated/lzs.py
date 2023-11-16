import collections

class BitReader:
    """
    Gets a string or a iterable of chars (also mmap)
    representing bytes (ord) and permits to extract
    bits one by one like a stream
    """

    def __init__(self, data_bytes):
        if False:
            for i in range(10):
                print('nop')
        self._bits = collections.deque()
        for byte in data_bytes:
            for n in range(8):
                self._bits.append(bool(byte >> 7 - n & 1))

    def getBit(self):
        if False:
            print('Hello World!')
        return self._bits.popleft()

    def getBits(self, num):
        if False:
            return 10
        res = 0
        for i in range(num):
            res += self.getBit() << num - 1 - i
        return res

    def getByte(self):
        if False:
            while True:
                i = 10
        return self.getBits(8)

    def __len__(self):
        if False:
            return 10
        return len(self._bits)

class RingList:
    """
    When the list is full, for every item appended
    the older is removed
    """

    def __init__(self, length):
        if False:
            i = 10
            return i + 15
        self.__data__ = collections.deque()
        self.__full__ = False
        self.__max__ = length

    def append(self, x):
        if False:
            while True:
                i = 10
        if self.__full__:
            self.__data__.popleft()
        self.__data__.append(x)
        if self.size() == self.__max__:
            self.__full__ = True

    def get(self):
        if False:
            print('Hello World!')
        return self.__data__

    def size(self):
        if False:
            while True:
                i = 10
        return len(self.__data__)

    def maxsize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__max__

    def __getitem__(self, n):
        if False:
            print('Hello World!')
        if n >= self.size():
            return None
        return self.__data__[n]

def LZSDecompress(data, window=RingList(2048)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets a string or a iterable of chars (also mmap)\n    representing bytes (ord) and an optional\n    pre-populated dictionary; return the decompressed\n    string and the final dictionary\n    '
    reader = BitReader(data)
    result = ''
    while True:
        bit = reader.getBit()
        if not bit:
            char = reader.getByte()
            result += chr(char)
            window.append(char)
        else:
            bit = reader.getBit()
            if bit:
                offset = reader.getBits(7)
                if offset == 0:
                    break
            else:
                offset = reader.getBits(11)
            lenField = reader.getBits(2)
            if lenField < 3:
                length = lenField + 2
            else:
                lenField <<= 2
                lenField += reader.getBits(2)
                if lenField < 15:
                    length = (lenField & 15) + 5
                else:
                    lenCounter = 0
                    lenField = reader.getBits(4)
                    while lenField == 15:
                        lenField = reader.getBits(4)
                        lenCounter += 1
                    length = 15 * lenCounter + 8 + lenField
            for i in range(length):
                char = window[-offset]
                result += chr(char)
                window.append(char)
    return (result, window)