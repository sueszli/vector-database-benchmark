from visidata import Sheet, ColumnItem, asyncthread, options

@VisiData.api
def open_tsv(vd, p):
    if False:
        i = 10
        return i + 15
    return MinimalTsvSheet(p.name, source=p)

class MinimalTsvSheet(Sheet):
    rowtype = 'rows'

    @asyncthread
    def reload(self):
        if False:
            while True:
                i = 10
        self.rows = []
        delim = options.delimiter
        header = True
        with open(self.source, encoding=options.encoding) as fp:
            for line in fp:
                line = line[:-1]
                if header:
                    if delim in line:
                        header = False
                        self.columns = []
                        for (i, colname) in enumerate(line.split()):
                            self.addColumn(ColumnItem(colname, i))
                    continue
                self.addRow(line.split(delim))
if __name__ == '__main__':
    import sys
    from visidata import run, Path
    run(*(open_tsv(Path(fn)) for fn in sys.argv[1:]))