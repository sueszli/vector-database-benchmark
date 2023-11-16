"""
Topic: 只允许关键字形式的参数
Desc : 
"""

def recv(maxsize, *, block):
    if False:
        while True:
            i = 10
    'Receives a message'
    pass
recv(1024, block=True)

def minimum(*values, clip=None):
    if False:
        return 10
    m = min(values)
    if clip is not None:
        m = clip if clip > m else m
    return m
minimum(1, 5, 2, -5, 10)
minimum(1, 5, 2, -5, 10, clip=0)