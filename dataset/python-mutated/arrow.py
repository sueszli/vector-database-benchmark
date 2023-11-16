from collections import defaultdict
from visidata import Sheet, VisiData, TypedWrapper, anytype, date, vlen, Column, vd

@VisiData.api
def open_arrow(vd, p):
    if False:
        i = 10
        return i + 15
    'Apache Arrow IPC file format'
    return ArrowSheet(p.name, source=p)

@VisiData.api
def open_arrows(vd, p):
    if False:
        for i in range(10):
            print('nop')
    'Apache Arrow IPC streaming format'
    return ArrowSheet(p.name, source=p)

def arrow_to_vdtype(t):
    if False:
        while True:
            i = 10
    pa = vd.importExternal('pyarrow')
    arrow_to_vd_typemap = {pa.lib.Type_BOOL: bool, pa.lib.Type_UINT8: int, pa.lib.Type_UINT16: int, pa.lib.Type_UINT32: int, pa.lib.Type_UINT64: int, pa.lib.Type_INT8: int, pa.lib.Type_INT16: int, pa.lib.Type_INT32: int, pa.lib.Type_INT64: int, pa.lib.Type_HALF_FLOAT: float, pa.lib.Type_FLOAT: float, pa.lib.Type_DOUBLE: float, pa.lib.Type_DATE32: date, pa.lib.Type_DATE64: date, pa.lib.Type_TIME32: date, pa.lib.Type_TIME64: date, pa.lib.Type_TIMESTAMP: date, pa.lib.Type_DURATION: int, pa.lib.Type_BINARY: bytes, pa.lib.Type_LARGE_BINARY: vlen}
    return arrow_to_vd_typemap.get(t.id, anytype)

class ArrowSheet(Sheet):

    def iterload(self):
        if False:
            print('Hello World!')
        pa = vd.importExternal('pyarrow')
        try:
            with pa.OSFile(str(self.source), 'rb') as fp:
                self.coldata = pa.ipc.open_file(fp).read_all()
        except pa.lib.ArrowInvalid as e:
            with pa.OSFile(str(self.source), 'rb') as fp:
                self.coldata = pa.ipc.open_stream(fp).read_all()
        self.columns = []
        for (colnum, col) in enumerate(self.coldata):
            coltype = arrow_to_vdtype(self.coldata.schema.types[colnum])
            colname = self.coldata.schema.names[colnum]
            self.addColumn(Column(colname, type=coltype, expr=colnum, getter=lambda c, r: c.sheet.coldata[c.expr][r[0]].as_py()))
        for rownum in range(max((len(c) for c in self.coldata))):
            yield [rownum]

@VisiData.api
def save_arrow(vd, p, sheet, streaming=False):
    if False:
        while True:
            i = 10
    pa = vd.importExternal('pyarrow')
    np = vd.importExternal('numpy')
    typemap = {anytype: pa.string(), int: pa.int64(), vlen: pa.int64(), float: pa.float64(), str: pa.string(), date: pa.date64()}
    for t in vd.numericTypes:
        if t not in typemap:
            typemap[t] = pa.float64()
    databycol = defaultdict(list)
    for typedvals in sheet.iterdispvals(format=False):
        for (col, val) in typedvals.items():
            if isinstance(val, TypedWrapper):
                val = None
            databycol[col].append(val)
    data = [pa.array(vals, type=typemap.get(col.type, pa.string())) for (col, vals) in databycol.items()]
    schema = pa.schema([(c.name, typemap.get(c.type, pa.string())) for c in sheet.visibleCols])
    with p.open_bytes(mode='w') as outf:
        if streaming:
            with pa.ipc.new_stream(outf, schema) as writer:
                writer.write_batch(pa.record_batch(data, names=[c.name for c in sheet.visibleCols]))
        else:
            with pa.ipc.new_file(outf, schema) as writer:
                writer.write_batch(pa.record_batch(data, names=[c.name for c in sheet.visibleCols]))

@VisiData.api
def save_arrows(vd, p, sheet):
    if False:
        print('Hello World!')
    return vd.save_arrow(p, sheet, streaming=True)