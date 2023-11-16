"""
(PDF 1.2) Decompresses data encoded using the zlib/deflate
compression method, reproducing the original text or binary
data.
"""
import copy
import typing
import zlib

class FlateDecode:
    """
    (PDF 1.2) Decompresses data encoded using the zlib/deflate
    compression method, reproducing the original text or binary
    data.
    """

    @staticmethod
    def decode(bytes_in: bytes, bits_per_component: int=8, columns: int=1, predictor: int=1) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Decompresses data encoded using the zlib/deflate\n        compression method\n        '
        if len(bytes_in) == 0:
            return bytes_in
        assert predictor in [1, 2, 10, 11, 12, 13, 14, 15], 'Illegal argument exception. predictor must be in [1, 2, 10, 11, 12, 13, 14, 15].'
        assert bits_per_component in [1, 2, 4, 8], 'Illegal argument exception. bits_per_component must be in [1, 2, 4, 8].'
        bytes_after_zlib = zlib.decompress(bytes_in, bufsize=4092)
        if predictor == 1:
            return bytes_after_zlib
        bytes_per_row: int = int((columns * bits_per_component + 7) / 8)
        bytes_per_pixel = int(bits_per_component / 8)
        current_row: typing.List[int] = [0 for _ in range(0, bytes_per_row)]
        prior_row: typing.List[int] = [0 for _ in range(0, bytes_per_row)]
        number_of_rows = int(len(bytes_after_zlib) / bytes_per_row)
        bytes_after_predictor = [int(x) for x in bytes_after_zlib]
        if predictor == 2:
            if bits_per_component == 8:
                for row in range(0, number_of_rows):
                    row_start_index = row * bytes_per_row
                    for col in range(1, bytes_per_row):
                        bytes_after_predictor[row_start_index + col] = (bytes_after_predictor[row_start_index + col] + bytes_after_predictor[row_start_index + col - 1]) % 256
                return bytes([int(x) % 256 for x in bytes_after_predictor])
        bytes_after_predictor = []
        pos = 0
        while pos + bytes_per_row <= len(bytes_after_zlib):
            filter_type = bytes_after_zlib[pos]
            pos += 1
            current_row = [x for x in bytes_after_zlib[pos:pos + bytes_per_row]]
            pos += bytes_per_row
            if filter_type == 0:
                pass
            if filter_type == 1:
                for i in range(bytes_per_pixel, bytes_per_row):
                    current_row[i] = (current_row[i] + current_row[i - bytes_per_pixel]) % 256
            if filter_type == 2:
                for i in range(0, bytes_per_row):
                    current_row[i] = (current_row[i] + prior_row[i]) % 256
            if filter_type == 3:
                for i in range(0, bytes_per_pixel):
                    current_row[i] += int(prior_row[i] / 2)
                for i in range(bytes_per_pixel, bytes_per_row):
                    current_row[i] += int((current_row[i - bytes_per_pixel] + prior_row[i]) / 2)
                    current_row[i] %= 256
            if filter_type == 4:
                for i in range(0, bytes_per_pixel):
                    current_row[i] += prior_row[i]
                for i in range(bytes_per_pixel, bytes_per_row):
                    a = current_row[i - bytes_per_pixel]
                    b = prior_row[i]
                    c = prior_row[i - bytes_per_pixel]
                    p = a + b - c
                    pa = abs(p - a)
                    pb = abs(p - b)
                    pc = abs(p - c)
                    ret = 0
                    if pa <= pb and pa <= pc:
                        ret = a
                    elif pb <= pc:
                        ret = b
                    else:
                        ret = c
                    current_row[i] = (current_row[i] + ret) % 256
            for i in range(0, len(current_row)):
                bytes_after_predictor.append(current_row[i])
            prior_row = copy.deepcopy(current_row)
        return bytes([int(x) % 256 for x in bytes_after_predictor])