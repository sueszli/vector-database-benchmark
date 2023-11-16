"""
Topic: 固定长度迭代文件
Desc : 
"""
from functools import partial

def iterate_fixed():
    if False:
        return 10
    RECORD_SIZE = 32
    with open('somefile.data', 'rb') as f:
        records = iter(partial(f.read, RECORD_SIZE), b'')
        for r in records:
            print(r)
if __name__ == '__main__':
    iterate_fixed()