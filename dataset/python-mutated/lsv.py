import collections
from visidata import VisiData, Sheet, ItemColumn

@VisiData.api
def open_lsv(vd, p):
    if False:
        while True:
            i = 10
    return LsvSheet(p.name, source=p)

@VisiData.api
def save_lsv(vd, p, *vsheets):
    if False:
        print('Hello World!')
    vs = vsheets[0]
    with p.open(mode='w', encoding=vs.options.save_encoding) as fp:
        for row in vs.iterrows():
            for col in vs.visibleCols:
                fp.write('%s: %s\n' % (col.name, col.getDisplayValue(row)))
            fp.write('\n')

class LsvSheet(Sheet):

    def addRow(self, row, **kwargs):
        if False:
            while True:
                i = 10
        super().addRow(row, **kwargs)
        for k in row:
            if k not in self._knownCols:
                self.addColumn(ItemColumn(k))
                self._knownCols.add(k)

    def iterload(self):
        if False:
            return 10
        self.columns = []
        self.rows = []
        self._knownCols = set()
        row = collections.defaultdict(str)
        k = ''
        for line in self.open_text_source():
            line = line.strip()
            if not line:
                yield row
                row = collections.defaultdict(str)
            if ':' in line:
                (k, line) = line.split(':', maxsplit=1)
            row[k.strip()] += line.strip()
        if row:
            yield row