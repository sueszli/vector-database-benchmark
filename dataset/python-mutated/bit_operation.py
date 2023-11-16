"""
Fundamental bit operation:
    get_bit(num, i): get an exact bit at specific index
    set_bit(num, i): set a bit at specific index
    clear_bit(num, i): clear a bit at specific index
    update_bit(num, i, bit): update a bit at specific index
"""
'\nThis function shifts 1 over by i bits, creating a value being like 0001000. By\nperforming an AND with num, we clear all bits other than the bit at bit i.\nFinally we compare that to 0\n'

def get_bit(num, i):
    if False:
        print('Hello World!')
    return num & 1 << i != 0
'\nThis function shifts 1 over by i bits, creating a value being like 0001000. By\nperforming an OR with num, only value at bit i will change.\n'

def set_bit(num, i):
    if False:
        return 10
    return num | 1 << i
'\nThis method operates in almost the reverse of set_bit\n'

def clear_bit(num, i):
    if False:
        while True:
            i = 10
    mask = ~(1 << i)
    return num & mask
'\nTo set the ith bit to value, we first clear the bit at position i by using a\nmask. Then, we shift the intended value. Finally we OR these two numbers\n'

def update_bit(num, i, bit):
    if False:
        i = 10
        return i + 15
    mask = ~(1 << i)
    return num & mask | bit << i