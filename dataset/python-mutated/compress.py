import zlib

def compress(data):
    if False:
        return 10
    ' Compress given data\n    :param str data: the data in string\n    :return str: string contained compressed data\n    '
    return zlib.compress(data)

def decompress(data):
    if False:
        return 10
    '\n    Decompress the data\n    :param str data: data to be decompressed\n    :return str: string containing uncompressed data\n    '
    return zlib.decompress(data)