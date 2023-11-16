"""
Topic: 文件的内存映射
Desc : 
"""
import os
import mmap

def memory_map(filename, access=mmap.ACCESS_WRITE):
    if False:
        while True:
            i = 10
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

def mem_mapping():
    if False:
        while True:
            i = 10
    pass
if __name__ == '__main__':
    mem_mapping()