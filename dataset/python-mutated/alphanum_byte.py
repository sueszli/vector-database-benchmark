from __future__ import division
from __future__ import absolute_import
from . import random_funcs
ALPHANUMERIC_BYTES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def alphanumeric_check(c):
    if False:
        while True:
            i = 10
    if type(c) == int:
        c = chr(c & 255)
    return c.isalnum()

def alphanumeric_get_byte():
    if False:
        print('Hello World!')
    return ord(random_funcs.randel(ALPHANUMERIC_BYTES))

def alphanumeric_get_byte_ltmax(max):
    if False:
        i = 10
        return i + 15
    sz = 0
    while sz < len(ALPHANUMERIC_BYTES) and ord(ALPHANUMERIC_BYTES[sz]) <= max:
        sz += 1
    return ord(random_funcs.randel(ALPHANUMERIC_BYTES[:sz]))

def off_gen(c):
    if False:
        i = 10
        return i + 15
    if c >= 0 and c <= 74:
        max = 16 * 7 + 10 - c
        while True:
            x = alphanumeric_get_byte_ltmax(max)
            if alphanumeric_check(c + x):
                return x
    return 0

def alphanumeric_get_complement(c):
    if False:
        i = 10
        return i + 15
    c &= 255
    while True:
        ret = alphanumeric_get_byte()
        if alphanumeric_check(c ^ ret):
            return ret