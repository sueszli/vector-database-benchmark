import time
import sys
import smart_open
from smart_open.bytebuffer import ByteBuffer

def raw_bytebuffer_benchmark():
    if False:
        print('Hello World!')
    buffer = ByteBuffer()
    start = time.time()
    for _ in range(10000):
        assert buffer.fill([b'X' * 1000]) == 1000
    return time.time() - start

def file_read_benchmark(filename):
    if False:
        print('Hello World!')
    file = smart_open.open(filename, mode='rb')
    start = time.time()
    read = file.read(100000000)
    end = time.time()
    if len(read) < 100000000:
        print('File smaller than 100MB')
    return end - start
print('Raw ByteBuffer benchmark:', raw_bytebuffer_benchmark())
if len(sys.argv) > 1:
    bench_result = file_read_benchmark(sys.argv[1])
    print('File read benchmark', bench_result)