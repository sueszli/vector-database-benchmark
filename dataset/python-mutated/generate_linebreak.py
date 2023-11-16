from __future__ import print_function
import re
breaking = 'OP    CL    CP    QU    GL    NS    EX    SY    IS    PR    PO    NU    AL    HL    ID    IN    HY    BA    BB    B2    ZW    CM    WJ    H2    H3    JL    JV    JT    RI\nOP    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    @    ^    ^    ^    ^    ^    ^    ^\nCL    _    ^    ^    %    %    ^    ^    ^    ^    %    %    _    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nCP    _    ^    ^    %    %    ^    ^    ^    ^    %    %    %    %    %    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nQU    ^    ^    ^    %    %    %    ^    ^    ^    %    %    %    %    %    %    %    %    %    %    %    ^    #    ^    %    %    %    %    %    %\nGL    %    ^    ^    %    %    %    ^    ^    ^    %    %    %    %    %    %    %    %    %    %    %    ^    #    ^    %    %    %    %    %    %\nNS    _    ^    ^    %    %    %    ^    ^    ^    _    _    _    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nEX    _    ^    ^    %    %    %    ^    ^    ^    _    _    _    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nSY    _    ^    ^    %    %    %    ^    ^    ^    _    _    %    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nIS    _    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nPR    %    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    %    _    %    %    _    _    ^    #    ^    %    %    %    %    %    _\nPO    %    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nNU    %    ^    ^    %    %    %    ^    ^    ^    %    %    %    %    %    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nAL    %    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nHL    %    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nID    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nIN    _    ^    ^    %    %    %    ^    ^    ^    _    _    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nHY    _    ^    ^    %    _    %    ^    ^    ^    _    _    %    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nBA    _    ^    ^    %    _    %    ^    ^    ^    _    _    _    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nBB    %    ^    ^    %    %    %    ^    ^    ^    %    %    %    %    %    %    %    %    %    %    %    ^    #    ^    %    %    %    %    %    %\nB2    _    ^    ^    %    %    %    ^    ^    ^    _    _    _    _    _    _    _    %    %    _    ^    ^    #    ^    _    _    _    _    _    _\nZW    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    _    ^    _    _    _    _    _    _    _    _\nCM    %    ^    ^    %    %    %    ^    ^    ^    _    _    %    %    %    _    %    %    %    _    _    ^    #    ^    _    _    _    _    _    _\nWJ    %    ^    ^    %    %    %    ^    ^    ^    %    %    %    %    %    %    %    %    %    %    %    ^    #    ^    %    %    %    %    %    %\nH2    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    %    %    _\nH3    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    _    %    _\nJL    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    %    %    %    %    _    _\nJV    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    %    %    _\nJT    _    ^    ^    %    %    %    ^    ^    ^    _    %    _    _    _    _    %    %    %    _    _    ^    #    ^    _    _    _    _    %    _\nRI    _    ^    ^    %    %    %    ^    ^    ^    _    _    _    _    _    _    _    %    %    _    _    ^    #    ^    _    _    _    _    _    %\n'
other_classes = ' PITCH AI BK CB CJ CR LF NL SA SG SP XX'
lines = breaking.split('\n')
print('# This is generated code. Do not edit.')
print()
cl = {}
for (i, j) in enumerate((lines[0] + other_classes).split()):
    print('cdef char BC_{} = {}'.format(j, i))
    cl[j] = i
print('CLASSES = {')
for (i, j) in enumerate((lines[0] + other_classes).split()):
    print('    "{}" : {},'.format(j, i))
    cl[j] = i
print('}')
rules = []
for l in lines[1:]:
    for c in l.split()[1:]:
        rules.append(c)
print()
print('cdef char *break_rules = "' + ''.join(rules) + '"')
cc = ['XX'] * 65536
for l in open('LineBreak.txt'):
    m = re.match('(\\w+)\\.\\.(\\w+);(\\w\\w)', l)
    if m:
        start = int(m.group(1), 16)
        end = int(m.group(2), 16)
        if start > 65535:
            continue
        if end > 65535:
            end = 65535
        for i in range(start, end + 1):
            cc[i] = m.group(3)
        continue
    m = re.match('(\\w+);(\\w\\w)', l)
    if m:
        start = int(m.group(1), 16)
        if start > 65535:
            continue
        cc[start] = m.group(2)
        continue

def generate(name, func):
    if False:
        while True:
            i = 10
    ncc = []
    for (i, ccl) in enumerate(cc):
        ncc.append(func(i, ccl))
    assert 'CJ' not in ncc
    assert 'AI' not in ncc
    print('cdef char *break_' + name + ' = "' + ''.join(('\\x%02x' % cl[i] for i in ncc)) + '"')

def western(i, cl):
    if False:
        print('Hello World!')
    if cl == 'CJ':
        return 'ID'
    elif cl == 'AI':
        return 'AL'
    return cl
hyphens = [8208, 8211, 12316, 12448]
iteration = [12293, 12347, 12445, 12446, 12541, 12542]
inseperable = [8229, 8230]
centered = [58, 59, 12539, 65306, 65307, 65381, 33, 63, 8252, 8263, 8264, 8265, 65281, 65311]
postfixes = [37, 162, 176, 8240, 8242, 8243, 8451, 65285, 65504]
prefixes = [36, 163, 165, 8364, 8470, 65284, 65505, 65509]

def cjk_strict(i, cl):
    if False:
        print('Hello World!')
    if cl == 'CJ':
        return 'NS'
    if cl == 'AI':
        return 'ID'
    return cl

def cjk_normal(i, cl):
    if False:
        return 10
    if i in hyphens:
        return 'ID'
    if cl == 'CJ':
        return 'ID'
    if cl == 'AI':
        return 'ID'
    return cl

def cjk_loose(i, cl):
    if False:
        while True:
            i = 10
    if i in hyphens:
        return 'ID'
    if i in iteration:
        return 'ID'
    if i in inseperable:
        return 'ID'
    if i in centered:
        return 'ID'
    if i in postfixes:
        return 'ID'
    if i in prefixes:
        return 'ID'
    if cl == 'CJ':
        return 'ID'
    if cl == 'AI':
        return 'ID'
    return cl
generate('western', western)
generate('cjk_strict', cjk_strict)
generate('cjk_normal', cjk_normal)
generate('cjk_loose', cjk_loose)