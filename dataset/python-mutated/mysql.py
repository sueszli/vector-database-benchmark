from contextlib import contextmanager
from urllib.parse import urlparse, unquote
from visidata import VisiData, vd, Sheet, anytype, asyncthread, ColumnItem

def codeToType(type_code, colname):
    if False:
        return 10
    import MySQLdb as mysql
    types = mysql.constants.FIELD_TYPE
    if type_code in (types.TINY, types.SHORT, types.LONG, types.LONGLONG, types.INT24):
        return int
    if type_code in (types.FLOAT, types.DOUBLE, types.DECIMAL, types.NEWDECIMAL):
        return float
    if type_code == mysql.STRING:
        return str
    return anytype

@VisiData.api
def openurl_mysql(vd, url, filetype=None):
    if False:
        print('Hello World!')
    url = urlparse(url.given)
    dbname = url.path[1:]
    return MyTablesSheet(dbname + '_tables', sql=SQL(url), schema=dbname)

class SQL:

    def __init__(self, url):
        if False:
            return 10
        self.url = url

    @contextmanager
    def cur(self, qstr):
        if False:
            while True:
                i = 10
        import MySQLdb as mysql
        import MySQLdb.cursors as cursors
        dbname = self.url.path[1:]
        connection = mysql.connect(user=self.url.username, database=self.url.path[1:], host=self.url.hostname, port=self.url.port or 3306, password=unquote(self.url.password), use_unicode=True, charset='utf8', cursorclass=cursors.SSCursor)
        try:
            cursor = connection.cursor()
            cursor.execute(qstr)
            with cursor as c:
                yield c
        finally:
            cursor.close()
            connection.close()

    @asyncthread
    def query_async(self, qstr, callback=None):
        if False:
            i = 10
            return i + 15
        with self.cur(qstr) as cur:
            callback(cur)

def cursorToColumns(cur, sheet):
    if False:
        while True:
            i = 10
    sheet.columns = []
    for (i, coldesc) in enumerate(cur.description):
        (name, type, *_) = coldesc
        sheet.addColumn(ColumnItem(name, i, type=codeToType(type, name)))

class MyTablesSheet(Sheet):
    rowtype = 'tables'

    def iterload(self):
        if False:
            print('Hello World!')
        qstr = f"\n            select\n                t.table_name,\n                column_count.ncols,\n                t.table_rows as est_nrows\n            from\n                information_schema.tables t,\n                (\n                    select\n                        table_name,\n                        count(column_name) as ncols\n                    from\n                        information_schema.columns\n                    where\n                        table_schema = '{self.schema}'\n                    group by\n                        table_name\n                ) as column_count\n            where\n                t.table_name = column_count.table_name\n                AND t.table_schema = '{self.schema}';\n        "
        with self.sql.cur(qstr) as cur:
            r = cur.fetchone()
            if r:
                yield r
            cursorToColumns(cur, self)
            self.setKeys(self.columns[0:1])
            for r in cur:
                yield r

    def openRow(self, row):
        if False:
            return 10
        return MyTable(self.name + '.' + row[0], source=row[0], sql=self.sql)

class MyTable(Sheet):

    def iterload(self):
        if False:
            print('Hello World!')
        with self.sql.cur('SELECT * FROM ' + self.source) as cur:
            r = cur.fetchone()
            if r is None:
                return
            yield r
            cursorToColumns(cur, self)
            while True:
                try:
                    r = cur.fetchone()
                    if r is None:
                        break
                    yield r
                except UnicodeDecodeError as e:
                    vd.exceptionCaught(e)