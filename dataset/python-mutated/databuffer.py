import struct
from .variables import LONG_STANDARD_SIZE

class DataBuffer:
    """ Data buffer that helps with network communication. """

    def __init__(self):
        if False:
            return 10
        ' Create new data buffer '
        self.buffered_data = b''

    def append_ulong(self, num):
        if False:
            return 10
        '\n        Append given number to data buffer written as unsigned long\n        in network order\n        :param long num: number to append (must be higher than 0)\n        '
        if num < 0:
            raise AttributeError('num must be grater than 0')
        bytes_num_rep = struct.pack('!L', num)
        self.buffered_data += bytes_num_rep
        return bytes_num_rep

    def append_bytes(self, data):
        if False:
            while True:
                i = 10
        ' Append given bytes to data buffer\n        :param bytes data: bytes to append\n        '
        self.buffered_data += data

    def data_size(self):
        if False:
            i = 10
            return i + 15
        ' Return size of data in buffer\n        :return int: size of data in buffer\n        '
        return len(self.buffered_data)

    def peek_ulong(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check long number that is located at the beginning of this data buffer\n        :return (long|None): number at the beginning of the buffer if it's there\n        "
        if len(self.buffered_data) < LONG_STANDARD_SIZE:
            return None
        (ret_val,) = struct.unpack('!L', self.buffered_data[0:LONG_STANDARD_SIZE])
        return ret_val

    def read_ulong(self):
        if False:
            print('Hello World!')
        '\n        Remove long number at the beginning of this data buffer and return it.\n        :return long: long number removed from the beginning of buffer\n        '
        val_ = self.peek_ulong()
        if val_ is None:
            raise ValueError('buffer_data is shorter than {}'.format(LONG_STANDARD_SIZE))
        self.buffered_data = self.buffered_data[LONG_STANDARD_SIZE:]
        return val_

    def peek_bytes(self, num_bytes):
        if False:
            i = 10
            return i + 15
        "\n        Return first <num_bytes> bytes from buffer. Doesn't change the buffer.\n        :param long num_bytes: how many bytes should be read from buffer\n        :return bytes: first <num_bytes> bytes from buffer\n        "
        if num_bytes > len(self.buffered_data):
            raise AttributeError('num_bytes is grater than buffer length')
        ret_bytes = self.buffered_data[:num_bytes]
        return ret_bytes

    def read_bytes(self, num_bytes):
        if False:
            return 10
        '\n        Remove first <num_bytes> bytes from buffer and return them.\n        :param long num_bytes: how many bytes should be read and removed\n         from buffer\n        :return bytes: bytes removed form buffer\n        '
        val_ = self.peek_bytes(num_bytes)
        self.buffered_data = self.buffered_data[num_bytes:]
        return val_

    def read_all(self):
        if False:
            while True:
                i = 10
        '\n        Return all data from buffer and clear the buffer.\n        :return bytes: all data that was in the buffer.\n        '
        ret_data = self.buffered_data
        self.buffered_data = b''
        return ret_data

    def read_len_prefixed_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read long number from the buffer and then read bytes with that length\n        from the buffer\n        :return bytes: first bytes from the buffer (after long)\n        '
        ret_bytes = None
        if self.data_size() >= LONG_STANDARD_SIZE and self.data_size() >= self.peek_ulong() + LONG_STANDARD_SIZE:
            num_bytes = self.read_ulong()
            ret_bytes = self.read_bytes(num_bytes)
        return ret_bytes

    def get_len_prefixed_bytes(self):
        if False:
            i = 10
            return i + 15
        '\n        Generator function that return from buffer datas preceded with\n        their length (long)\n        '
        while self.data_size() > LONG_STANDARD_SIZE and self.data_size() >= self.peek_ulong() + LONG_STANDARD_SIZE:
            num_bytes = self.read_ulong()
            yield self.read_bytes(num_bytes)

    def append_len_prefixed_bytes(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append length of a given data and then given data to the buffer\n        :param bytes data: data to append\n        '
        self.append_ulong(len(data))
        self.append_bytes(data)

    def clear_buffer(self):
        if False:
            while True:
                i = 10
        ' Remove all data from the buffer '
        self.buffered_data = b''