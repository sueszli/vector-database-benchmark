import sqlite3

class Point:

    def __init__(self, x, y):
        if False:
            print('Hello World!')
        (self.x, self.y) = (x, y)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(%f;%f)' % (self.x, self.y)

def adapt_point(point):
    if False:
        return 10
    return ('%f;%f' % (point.x, point.y)).encode('ascii')

def convert_point(s):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = list(map(float, s.split(b';')))
    return Point(x, y)
sqlite3.register_adapter(Point, adapt_point)
sqlite3.register_converter('point', convert_point)
p = Point(4.0, -3.2)
con = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute('create table test(p point)')
cur.execute('insert into test(p) values (?)', (p,))
cur.execute('select p from test')
print('with declared types:', cur.fetchone()[0])
cur.close()
con.close()
con = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_COLNAMES)
cur = con.cursor()
cur.execute('create table test(p)')
cur.execute('insert into test(p) values (?)', (p,))
cur.execute('select p as "p [point]" from test')
print('with column names:', cur.fetchone()[0])
cur.close()
con.close()