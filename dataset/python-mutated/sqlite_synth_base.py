"""
SQLite benchmark.

The goal of the benchmark is to test CFFI performance and going back and forth
between SQLite and Python a lot. Therefore the queries themselves are really
simple.
"""
import sqlite3
import math

class AvgLength(object):

    def __init__(self):
        if False:
            return 10
        self.sum = 0
        self.count = 0

    def step(self, x):
        if False:
            print('Hello World!')
        if x is not None:
            self.count += 1
            self.sum += len(x)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        return self.sum / float(self.count)

def bench_sqlite(loops):
    if False:
        return 10
    conn = sqlite3.connect(':memory:')
    conn.execute('create table cos (x, y, z);')
    for i in range(loops):
        cos_i = math.cos(i)
        conn.execute('insert into cos values (?, ?, ?)', [i, cos_i, str(i)])
    conn.create_function('cos', 1, math.cos)
    for (x, cosx1, cosx2) in conn.execute('select x, cos(x), y from cos'):
        assert math.cos(x) == cosx1 == cosx2
    conn.create_aggregate('avglength', 1, AvgLength)
    cursor = conn.execute('select avglength(z) from cos;')
    cursor.fetchone()[0]
    conn.execute('delete from cos;')
    conn.close()

def run_benchmark():
    if False:
        i = 10
        return i + 15
    bench_sqlite(1)
if __name__ == '__main__':
    run_benchmark()