"""
Topic: 编码/解码Base64
Desc : 
"""
import base64

def codec_base64():
    if False:
        return 10
    s = b'hello'
    a = base64.b64encode(s)
    print(a)
    b = base64.b64decode(a)
    print(b)
if __name__ == '__main__':
    codec_base64()