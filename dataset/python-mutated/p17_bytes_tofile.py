"""
Topic: 在文本模式文件中写入字节
Desc : 
"""
import sys

def bytes_tofile():
    if False:
        return 10
    sys.stdout.buffer.write(b'Hello\n')
if __name__ == '__main__':
    bytes_tofile()