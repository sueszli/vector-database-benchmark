import io

class CatchZeroByteWriter(io.BufferedWriter):
    """File handle to intercept 0-byte writes."""

    def write(self, buffer):
        if False:
            while True:
                i = 10
        nbytes = super().write(buffer)
        if nbytes == 0:
            raise ValueError('This writer does not allow empty writes')
        return nbytes