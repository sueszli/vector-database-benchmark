from visidata import vd, VisiData, Sheet, asyncthread, Progress, Column

@VisiData.api
class TransposeSheet(Sheet):

    def beforeLoad(self):
        if False:
            print('Hello World!')
        col = Column('_'.join((c.name for c in self.source.keyCols)), getter=lambda c, origcol: origcol.name)
        col.recalc(self)
        self.columns = [col]
        self.setKeys(self.columns)

    def loader(self):
        if False:
            while True:
                i = 10
        for row in Progress(self.source.rows, 'transposing'):
            self.addColumn(Column('_'.join(map(str, self.source.rowkey(row))), getter=lambda c, origcol, row=row: origcol.getValue(row)))
        self.rows = list(self.source.nonKeyVisibleCols)
Sheet.addCommand('T', 'transpose', 'vd.push(TransposeSheet(name+"_T", source=sheet))', 'open new sheet with rows and columns transposed')
vd.addMenuItems('Data > Transpose > transpose')