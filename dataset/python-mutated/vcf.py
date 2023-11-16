from visidata import VisiData, Column, getitemdef, PythonSheet, asyncthread, vd

@VisiData.api
def open_vcf(vd, p):
    if False:
        while True:
            i = 10
    return VcfSheet(p.name, source=p)

def unbox(col, row):
    if False:
        while True:
            i = 10
    v = getitemdef(row, col.expr)
    if not v:
        return None
    if len(v) == 1:
        return v[0].value
    return v

class VcfSheet(PythonSheet):
    rowtype = 'cards'

    @asyncthread
    def reload(self):
        if False:
            for i in range(10):
                print('nop')
        vobject = vd.importExternal('vobject')
        self.rows = []
        self.columns = []
        addedCols = set()
        lines = []
        for line in self.open_text_source():
            lines.append(line)
            if line.startswith('END:'):
                row = vobject.readOne('\n'.join(lines))
                for (k, v) in row.contents.items():
                    if v and str(v[0].value).startswith('(None)'):
                        continue
                    if not k in addedCols:
                        addedCols.add(k)
                        self.addColumn(Column(k, expr=k, getter=unbox))
                self.addRow(row.contents)
                lines = []