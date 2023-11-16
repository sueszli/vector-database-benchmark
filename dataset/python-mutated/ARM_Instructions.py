from __future__ import division
EOR = 1
SUB = 2
RSB = 3
MI = 4
PL = 5
LDR = 6
STR = 7
LDM = 8
STM = 9
ROR = 10
LSR = 11

def dpimm(op, cond, s, d, n, imm):
    if False:
        print('Hello World!')
    if type(imm) == int:
        x = chr(imm & 255)
    else:
        x = imm
    x += chr(d << 4 & 255)
    if s:
        if op == EOR:
            x += chr(48 | n)
        if op == SUB:
            x += chr(80 | n)
        if op == RSB:
            x += chr(112 | n)
    else:
        if op == SUB:
            x += chr(64 | n)
        if op == RSB:
            x += chr(96 | n)
    if cond == PL:
        x += 'R'
    else:
        x += 'B'
    return x

def dpshiftimm(op, s, d, n, a, imm):
    if False:
        return 10
    x = chr(96 | a)
    x += chr((d << 4 | imm >> 1) & 255)
    if s:
        if op == EOR:
            x += chr(48 | n)
        if op == SUB:
            x += chr(80 | n)
        if op == RSB:
            x += chr(112 | n)
    else:
        if op == SUB:
            x += chr(64 | n)
        if op == RSB:
            x += chr(96 | n)
    return x + 'P'

def dpshiftreg(op, s, d, n, a, shift, b):
    if False:
        print('Hello World!')
    x = ''
    if shift == LSR:
        x += chr(48 | a)
    else:
        x += chr(112 | a)
    x += chr((d << 4 | b) & 255)
    if s != 0:
        if op == EOR:
            x += chr(48 | n)
        if op == SUB:
            x += chr(80 | n)
        if op == RSB:
            x += chr(112 | n)
    else:
        if op == SUB:
            x += chr(64 | n)
        if op == RSB:
            x += chr(96 | n)
    return x + 'P'

def lsbyte(op, cond, d, n, imm):
    if False:
        print('Hello World!')
    if type(imm) == int:
        x = chr(imm & 255)
    else:
        x = imm
    x += chr(d << 4 & 255)
    if op == STR:
        x += chr(64 | n)
    else:
        x += chr(80 | n)
    if cond == PL:
        x += 'U'
    else:
        x += 'E'
    return x

def smul(d, reglH, reglL):
    if False:
        return 10
    return chr(reglL) + chr(reglH) + chr(64 | d) + 'Y'

def lmul(n, reglH, reglL):
    if False:
        while True:
            i = 10
    return chr(reglL) + chr(reglH) + chr(48 | n) + 'Y'

def swi(cond):
    if False:
        while True:
            i = 10
    x = '\x02\x00\x9f'
    if cond == MI:
        x += 'O'
    else:
        x += '_'
    return x

def bmi():
    if False:
        for i in range(10):
            print('nop')
    return 'ôÿÿK'

def sbyteposti(d, n, m, imm):
    if False:
        print('Hello World!')
    x = chr(96 | m)
    x += chr((d << 4 | imm >> 1) & 255)
    x += chr(64 | n)
    x += 'V'
    return x