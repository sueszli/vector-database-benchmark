"""
Topic: 读写二进制数组结构的数据
Desc : 
"""
from struct import Struct
from collections import namedtuple

def write_records(records, format, f):
    if False:
        i = 10
        return i + 15
    '\n    Write a sequence of tuples to a binary file of structures.\n    '
    record_struct = Struct(format)
    for r in records:
        f.write(record_struct.pack(*r))

def read_records(format, f):
    if False:
        for i in range(10):
            print('nop')
    record_struct = Struct(format)
    chunks = iter(lambda : f.read(record_struct.size), b'')
    return (record_struct.unpack(chunk) for chunk in chunks)

def unpack_records(format, data):
    if False:
        return 10
    record_struct = Struct(format)
    return (record_struct.unpack_from(data, offset) for offset in range(0, len(data), record_struct.size))
if __name__ == '__main__':
    records = [(1, 2.3, 4.5), (6, 7.8, 9.0), (12, 13.4, 56.7)]
    with open('data.b', 'wb') as f:
        write_records(records, '<idd', f)
    with open('data.b', 'rb') as f:
        for rec in read_records('<idd', f):
            pass
    with open('data.b', 'rb') as f:
        data = f.read()
    for rec in unpack_records('<idd', data):
        pass
    record_struct = Struct('<idd')
    print(record_struct.size)
    a = record_struct.pack(1, 2.0, 3.0)
    print(a)
    print(record_struct.unpack(a))
    Record = namedtuple('Record', ['kind', 'x', 'y'])
    with open('data.p', 'rb') as f:
        records = (Record(*r) for r in read_records('<idd', f))
    for r in records:
        print(r.kind, r.x, r.y)