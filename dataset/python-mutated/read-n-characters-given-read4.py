def read4(buf):
    if False:
        return 10
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

    def read(self, buf, n):
        if False:
            i = 10
            return i + 15
        '\n        :type buf: Destination buffer (List[str])\n        :type n: Maximum number of characters to read (int)\n        :rtype: The number of characters read (int)\n        '
        read_bytes = 0
        buffer = [''] * 4
        for i in xrange((n + 4 - 1) // 4):
            size = min(read4(buffer), n - read_bytes)
            buf[read_bytes:read_bytes + size] = buffer[:size]
            read_bytes += size
        return read_bytes