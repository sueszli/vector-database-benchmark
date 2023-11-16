import struct
from gzip import GzipFile
from io import BytesIO
from typing import List
from scrapy.http import Response

def gunzip(data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Gunzip the given data and return as much data as possible.\n\n    This is resilient to CRC checksum errors.\n    '
    f = GzipFile(fileobj=BytesIO(data))
    output_list: List[bytes] = []
    chunk = b'.'
    while chunk:
        try:
            chunk = f.read1(8196)
            output_list.append(chunk)
        except (OSError, EOFError, struct.error):
            if output_list:
                break
            raise
    return b''.join(output_list)

def gzip_magic_number(response: Response) -> bool:
    if False:
        return 10
    return response.body[:3] == b'\x1f\x8b\x08'