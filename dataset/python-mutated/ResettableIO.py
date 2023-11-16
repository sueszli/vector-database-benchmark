import io

class ResettableIO(io.RawIOBase):
    """
    Raw I/O implementation the input and output stream is resettable.
    """

    def set_input_bytes(self, b):
        if False:
            print('Hello World!')
        self._input_bytes = b
        self._input_offset = 0
        self._size = len(b)

    def readinto(self, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read up to len(b) bytes into the writable buffer *b* and return\n        the number of bytes read. If no bytes are available, None is returned.\n        '
        output_buffer_len = len(b)
        remaining = self._size - self._input_offset
        if remaining >= output_buffer_len:
            b[:] = self._input_bytes[self._input_offset:self._input_offset + output_buffer_len]
            self._input_offset += output_buffer_len
            return output_buffer_len
        elif remaining > 0:
            b[:remaining] = self._input_bytes[self._input_offset:self._input_offset + remaining]
            self._input_offset = self._size
            return remaining
        else:
            return None

    def set_output_stream(self, output_stream):
        if False:
            print('Hello World!')
        self._output_stream = output_stream

    def write(self, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write the given bytes or pyarrow.Buffer object *b* to the underlying\n        output stream and return the number of bytes written.\n        '
        if isinstance(b, bytes):
            self._output_stream.write(b)
        else:
            self._output_stream.write(b.to_pybytes())
        return len(b)

    def seekable(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def readable(self):
        if False:
            print('Hello World!')
        return self._size - self._input_offset

    def writable(self):
        if False:
            print('Hello World!')
        return True