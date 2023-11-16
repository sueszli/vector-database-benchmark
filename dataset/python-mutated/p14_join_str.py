"""
Topic: 字符串合并
Desc : 
"""

def combine(source, maxsize):
    if False:
        i = 10
        return i + 15
    parts = []
    size = 0
    for part in source:
        parts.append(part)
        size += len(part)
        if size > maxsize:
            yield ''.join(parts)
            parts = []
            size = 0
    yield ''.join(parts)

def sample():
    if False:
        print('Hello World!')
    yield 'Is'
    yield 'Chicago'
    yield 'Not'
    yield 'Chicago?'

def join_str():
    if False:
        i = 10
        return i + 15
    parts = ['Is', 'Chicago', 'Not', 'Chicago?']
    print(' '.join(parts))
    print(','.join(parts))
    print(''.join(parts))
    a = 'Is Chicago'
    b = 'Not Chicago?'
    c = 'ccc'
    print(a + ' ' + b)
    data = ['ACME', 50, 91.1]
    print(','.join((str(d) for d in data)))
    print(a + ':' + b + ':' + c)
    print(':'.join([a, b, c]))
    print(a, b, c, sep=':')
    for part in combine(sample(), 32768):
        print(part)
if __name__ == '__main__':
    join_str()