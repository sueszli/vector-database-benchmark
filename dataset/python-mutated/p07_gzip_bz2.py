"""
Topic: 读写压缩文件
Desc : 
"""
import gzip
import bz2

def gzip_bz2():
    if False:
        for i in range(10):
            print('nop')
    with gzip.open('somefile.gz', 'rt') as f:
        text = f.read()
    with bz2.open('somefile.bz2', 'rt') as f:
        text = f.read()
    with gzip.open('somefile.gz', 'wt') as f:
        f.write(text)
    with bz2.open('somefile.bz2', 'wt') as f:
        f.write(text)
    with gzip.open('somefile.gz', 'wt', compresslevel=5) as f:
        f.write(text)
    f = open('somefile.gz', 'rb')
    with gzip.open(f, 'rt') as g:
        text = g.read()
if __name__ == '__main__':
    gzip_bz2()