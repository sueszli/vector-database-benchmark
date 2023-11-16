import sqlite3

class IterChars:

    def __init__(self):
        if False:
            return 10
        self.count = ord('a')

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.count > ord('z'):
            raise StopIteration
        self.count += 1
        return (chr(self.count - 1),)
con = sqlite3.connect(':memory:')
cur = con.cursor()
cur.execute('create table characters(c)')
theIter = IterChars()
cur.executemany('insert into characters(c) values (?)', theIter)
cur.execute('select c from characters')
print(cur.fetchall())
con.close()