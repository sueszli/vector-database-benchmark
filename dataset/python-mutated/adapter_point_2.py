import sqlite3

class Point:

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (self.x, self.y) = (x, y)

def adapt_point(point):
    if False:
        i = 10
        return i + 15
    return '%f;%f' % (point.x, point.y)
sqlite3.register_adapter(Point, adapt_point)
con = sqlite3.connect(':memory:')
cur = con.cursor()
p = Point(4.0, -3.2)
cur.execute('select ?', (p,))
print(cur.fetchone()[0])
con.close()