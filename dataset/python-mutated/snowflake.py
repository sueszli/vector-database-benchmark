import time
from visidata import vd, BaseException, VisiData
from ._ibis import IbisTableSheet, IbisConnectionPool, IbisTableIndexSheet

@VisiData.api
def openurl_snowflake(vd, p, filetype=None):
    if False:
        return 10
    return IbisTableIndexSheet(p.name, source=p, filetype=None, database_name=None, ibis_conpool=IbisConnectionPool(p), sheet_type=SnowflakeSheet)

class SnowflakeSheet(IbisTableSheet):

    @property
    def countRows(self):
        if False:
            i = 10
            return i + 15
        r = super().countRows
        if r is None and self.cursor is None:
            return None
        return r

    def executeSql(self, sql):
        if False:
            i = 10
            return i + 15
        assert self.cursor is None
        with self.con as con:
            con = con.con
            if self.warehouse:
                con.execute(f'USE WAREHOUSE {self.warehouse}')
            with con.begin() as c:
                snowflake_conn = c.connection.dbapi_connection
                cursor = self.cursor = snowflake_conn.cursor()
                cursor.execute_async(sql)
                while snowflake_conn.is_still_running(snowflake_conn.get_query_status(cursor.sfqid)):
                    time.sleep(0.1)
        cursor.get_results_from_sfqid(cursor.sfqid)
        yield from cursor.fetchall()
        self.cursor = None

    def iterload(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            with self.con as con:
                if self.query is None:
                    self.query = self.baseQuery(con)
                yield from self.executeSql(self.ibis_to_sql(self.withRowcount(self.baseQuery(con))))
        except BaseException:
            if self.cursor:
                self.cancelQuery(self.cursor.sfqid)
            raise
        self.reloadColumns(self.query)

    def cancelQuery(self, qid):
        if False:
            return 10
        vd.status(f'canceling "{qid}"')
        with self.con as con:
            with con.begin() as con:
                cursor = con.connection.dbapi_connection.cursor()
                cursor.execute(f"SELECT SYSTEM$CANCEL_QUERY('{qid}')")
                vd.status(cursor.fetchall())
SnowflakeSheet.init('cursor', lambda : None)
SnowflakeSheet.init('warehouse', str)