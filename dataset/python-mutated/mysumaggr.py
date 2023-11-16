import sqlite3

class MySum:

    def __init__(self):
        if False:
            print('Hello World!')
        self.count = 0

    def step(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.count += value

    def finalize(self):
        if False:
            i = 10
            return i + 15
        return self.count
con = sqlite3.connect(':memory:')
con.create_aggregate('mysum', 1, MySum)
cur = con.cursor()
cur.execute('create table test(i)')
cur.execute('insert into test(i) values (1)')
cur.execute('insert into test(i) values (2)')
cur.execute('select mysum(i) from test')
print(cur.fetchone()[0])
con.close()