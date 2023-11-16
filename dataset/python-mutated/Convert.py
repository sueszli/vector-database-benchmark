import json
import numpy as np
import itertools

def _unzip(st, is_shorten=True):
    if False:
        i = 10
        return i + 15
    lines = st.split('/')
    V = []
    for l in lines:
        nl = []
        t = l.split('.')
        if len(t) > 1:
            nl.extend(_FromRepeatSt(t[0]) * [0])
            cs = list(t[1])
        else:
            cs = list(t[0])
        for c in cs:
            v = _FromZip(c)
            nl.append(int(v / 100 * 255))
        if len(nl) > 0:
            V.append(nl)
    code = [[' .' if v == 0 else ' ' + chr(ord('A') + v - 1) if v < 25 else chr(ord('p') + (v - 25) // 24) + chr(ord('A') + (v - 25) % 24) for v in row] for row in V]
    if is_shorten:
        rle = [[(len(list(g)), k.strip()) for (k, g) in itertools.groupby(row)] for row in code]
        for row in rle:
            if row[-1][1] == '.':
                row.pop()
        st = '$'.join((''.join([(str(n) if n > 1 else '') + k for (n, k) in row]) for row in rle)) + '!'
    else:
        st = '$'.join((''.join(row) for row in code)) + '!'
    return st

def _FromZip(c):
    if False:
        while True:
            i = 10
    return 0 if c == '0' else 100 if c == '1' else ord(c) - (192 - 1)

def _IsZip(c):
    if False:
        i = 10
        return i + 15
    return ord(c) >= 192

def _FromRepeatSt(st):
    if False:
        return 10
    return 1 if st == '' else _FromZip(st) if len(st) == 1 else _FromZip(st[0]) * 100 + _FromZip(st[1])
with open('animals.json', encoding='utf-8') as file:
    data = json.load(file)
for d in data:
    if 'name' in d:
        print(d['name'])
    if 'params' in d:
        d['params']['T'] = int(1 / d['params']['dt'])
        d['params'] = {k: d['params'][k] for k in ('R', 'T', 'b', 'm', 's', 'kn', 'gn')}
st = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
st = st.replace('},{', '},\n{') + '\n'
with open('animals2.json', 'w', encoding='utf-8') as file:
    file.write(st)