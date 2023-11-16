import sqlite3

class Point:

    def __init__(self, x, y):
        if False:
            i = 10
            return i + 15
        (self.x, self.y) = (x, y)

    def __conform__(self, protocol):
        if False:
            print('Hello World!')
        if protocol is sqlite3.PrepareProtocol:
            return '%f;%f' % (self.x, self.y)
con = sqlite3.connect(':memory:')
cur = con.cursor()
p = Point(4.0, -3.2)
cur.execute('select ?', (p,))
print(cur.fetchone()[0])
con.close()