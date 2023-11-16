def read4(buf):
    if False:
        i = 10
        return i + 15
    global file_content
    i = 0
    while i < len(file_content) and i < 4:
        buf[i] = file_content[i]
        i += 1
    if len(file_content) > 4:
        file_content = file_content[4:]
    else:
        file_content = ''
    return i

class Solution(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__buf4 = [''] * 4
        self.__i4 = 0
        self.__n4 = 0

    def read(self, buf, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type buf: Destination buffer (List[str])\n        :type n: Maximum number of characters to read (int)\n        :rtype: The number of characters read (int)\n        '
        i = 0
        while i < n:
            if self.__i4 < self.__n4:
                buf[i] = self.__buf4[self.__i4]
                i += 1
                self.__i4 += 1
            else:
                self.__n4 = read4(self.__buf4)
                if self.__n4:
                    self.__i4 = 0
                else:
                    break
        return i