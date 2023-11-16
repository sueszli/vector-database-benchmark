"""
Decompresses data encoded using a byte-oriented run-length
encoding algorithm, reproducing the original text or binary data
(typically monochrome image data, or any data that contains
frequent long runs of a single byte value).
"""
import logging
logger = logging.getLogger(__name__)

class RunLengthDecode:
    """
    Decompresses data encoded using a byte-oriented run-length
    encoding algorithm, reproducing the original text or binary data
    (typically monochrome image data, or any data that contains
    frequent long runs of a single byte value).
    """

    @staticmethod
    def decode(bytes_in: bytes) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Decompresses data encoded using a byte-oriented run-length\n        encoding algorithm\n        '
        if len(bytes_in) == 0:
            return bytes_in
        bytes_out = bytearray()
        i: int = 0
        while i < len(bytes_in):
            b = bytes_in[i]
            if b == 128:
                break
            length: int = 0
            if 0 <= b <= 127:
                length = b + 1
                i += 1
                for j in range(0, length):
                    bytes_out.append(bytes_in[i + j])
                i += length
                continue
            if 129 <= b <= 255:
                length = 257 - b
                i += 1
                for _ in range(0, length):
                    bytes_out.append(bytes_in[i])
                i += 1
        return bytes(bytes_out)