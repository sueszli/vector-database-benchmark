"""
Topic: 读写二进制文件
Desc : 
"""

def rw_binary():
    if False:
        return 10
    with open('somefile.bin', 'rb') as f:
        data = f.read()
    with open('somefile.bin', 'wb') as f:
        f.write(b'Hello World')
    t = 'Hello World'
    print(t[0])
    b = b'Hello World'
    print(b[0])
    for c in b:
        print(c)
if __name__ == '__main__':
    rw_binary()