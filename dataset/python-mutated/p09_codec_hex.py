"""
Topic: 编码/解码十六进制原始字符串
Desc : 
"""
import binascii
import base64

def codec_hex():
    if False:
        for i in range(10):
            print('nop')
    s = b'hello'
    h = binascii.b2a_hex(s)
    print(h)
    h = binascii.a2b_hex(h)
    print(h)
    h = base64.b16encode(s)
    print(h)
    print(h.decode('ascii'))
    h = base64.b16decode(h)
    print(h)
    print(h.decode('ascii'))
if __name__ == '__main__':
    codec_hex()