import os
from visidata import VisiData

@VisiData.api
def ansi(*args):
    if False:
        print('Hello World!')
    os.write(1, b'\x1b' + b''.join([str(x).encode('utf-8') for x in args]))

@VisiData.api
def set_titlebar(vd, title: str):
    if False:
        while True:
            i = 10
    ansi(']2;', title, '\x07')