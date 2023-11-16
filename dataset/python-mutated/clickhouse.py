import time
from visidata import BaseException, vd, VisiData, Progress
from ._ibis import IbisTableSheet, IbisTableIndexSheet, IbisConnectionPool

@VisiData.api
def openurl_clickhouse(vd, p, filetype=None):
    if False:
        print('Hello World!')
    vd.configure_ibis()
    return IbisTableIndexSheet(p.name, source=p, filetype=None, database_name=None, ibis_conpool=IbisConnectionPool(p), sheet_type=ClickhouseSheet)
vd.openurl_clickhouses = vd.openurl_clickhouse

class ClickhouseSheet(IbisTableSheet):

    @property
    def countRows(self):
        if False:
            print('Hello World!')
        if self.total_rows is not None:
            return self.total_rows
        return super().countRows

    def iterload(self):
        if False:
            return 10
        with self.con as con:
            qid = None
            try:
                if self.query is None:
                    self.query = self.baseQuery(con)
                self.reloadColumns(self.query, start=0)
                sqlstr = con.compile(self.query.limit(self.options.ibis_limit or None))
                with Progress(gerund='clickhousing', sheet=self) as prog:
                    settings = {'max_block_size': 10000}
                    with con.con.query_rows_stream(sqlstr, settings) as stream:
                        prog.total = int(stream.source.summary['total_rows_to_read'])
                        prog.made = 0
                        for row in stream:
                            prog.made += 1
                            yield row
                        self.total_rows = prog.total
            except Exception as e:
                raise
            except BaseException:
                if qid:
                    con.con.cancel(qid)
ClickhouseSheet.init('total_rows', lambda : None)