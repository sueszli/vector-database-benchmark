from visidata import VisiData, vd, Sheet, Column, Progress, SequenceSheet
vd.option('fixed_rows', 1000, 'number of rows to check for fixed width columns')
vd.option('fixed_maxcols', 0, 'max number of fixed-width columns to create (0 is no max)')

@VisiData.api
def open_fixed(vd, p):
    if False:
        while True:
            i = 10
    return FixedWidthColumnsSheet(p.name, source=p, headerlines=[])

class FixedWidthColumn(Column):

    def __init__(self, name, i, j, **kwargs):
        if False:
            return 10
        super().__init__(name, **kwargs)
        (self.i, self.j) = (i, j)

    def calcValue(self, row):
        if False:
            i = 10
            return i + 15
        return row[0][self.i:self.j]

    def putValue(self, row, value):
        if False:
            i = 10
            return i + 15
        value = str(value)[:self.j - self.i]
        j = self.j or len(row)
        row[0] = row[0][:self.i] + '%-*s' % (j - self.i, value) + row[0][self.j:]

def columnize(rows):
    if False:
        while True:
            i = 10
    'Generate (i,j) indexes for fixed-width columns found in rows'
    allNonspaces = set()
    for r in rows:
        for (i, ch) in enumerate(r):
            if not ch.isspace():
                allNonspaces.add(i)
    colstart = 0
    prev = 0
    for i in allNonspaces:
        if i > prev + 1:
            yield (colstart, i)
            colstart = i
        prev = i
    yield (colstart, prev + 1)

class FixedWidthColumnsSheet(SequenceSheet):
    rowtype = 'lines'

    def addRow(self, row, index=None):
        if False:
            while True:
                i = 10
        Sheet.addRow(self, row, index=index)

    def iterload(self):
        if False:
            while True:
                i = 10
        itsource = iter(self.source)
        maxcols = self.options.fixed_maxcols
        self.columns = []
        fixedRows = list(([x] for x in self.optlines(itsource, 'fixed_rows')))
        for (i, j) in columnize(list((r[0] for r in fixedRows))):
            if maxcols and self.nCols >= maxcols - 1:
                self.addColumn(FixedWidthColumn('', i, None))
                break
            else:
                self.addColumn(FixedWidthColumn('', i, j))
        yield from fixedRows
        self.setColNames(self.headerlines)
        yield from ([line] for line in itsource)

    def setCols(self, headerlines):
        if False:
            for i in range(10):
                print('nop')
        self.headerlines = headerlines

@VisiData.api
def save_fixed(vd, p, *vsheets):
    if False:
        for i in range(10):
            print('nop')
    with p.open(mode='w', encoding=vsheets[0].options.save_encoding) as fp:
        for sheet in vsheets:
            if len(vsheets) > 1:
                fp.write('%s\n\n' % sheet.name)
            widths = {}
            for col in Progress(sheet.visibleCols, gerund='sizing'):
                widths[col] = col.getMaxWidth(sheet.rows)
                fp.write(('{0:%s} ' % widths[col]).format(col.name))
            fp.write('\n')
            with Progress(gerund='saving'):
                for dispvals in sheet.iterdispvals(format=True):
                    for (col, val) in dispvals.items():
                        fp.write(('{0:%s%s.%s} ' % ('>' if vd.isNumeric(col) else '<', widths[col], widths[col])).format(val))
                    fp.write('\n')
            vd.status('%s save finished' % p)