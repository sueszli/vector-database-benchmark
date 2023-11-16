from visidata import *
from functools import wraps, partial

@asyncthread
@Sheet.api
def calcRowIndex(sheet, indexes):
    if False:
        return 10
    for (rownum, r) in enumerate(sheet.rows):
        indexes[sheet.rowid(r)] = rownum

@Sheet.lazy_property
def _rowindex(sheet):
    if False:
        i = 10
        return i + 15
    ret = {}
    sheet.calcRowIndex(ret)
    return ret

@Sheet.api
def rowindex(sheet, row):
    if False:
        for i in range(10):
            print('nop')
    'Returns the rowindex given the row.  May spawn a thread to compute underlying _rowindex.'
    return sheet._rowindex.get(sheet.rowid(row))

@Sheet.api
def prev(sheet, row):
    if False:
        print('Hello World!')
    'Return the row previous to the given row.'
    rownum = max(sheet.rowindex(row) - 1, 0)
    return LazyComputeRow(sheet, sheet.rows[rownum])

@Sheet.api
def addcol_rowindex(sheet, newcol):
    if False:
        for i in range(10):
            print('nop')
    oldAddRow = sheet.addRow

    def rownum_addRow(sheet, col, row, index=None):
        if False:
            i = 10
            return i + 15
        if index is None:
            index = len(sheet.rows)
        col._rowindex[sheet.rowid(row)] = index
        return oldAddRow(row, index)
    sheet.addRow = wraps(oldAddRow)(partial(rownum_addRow, sheet, newcol))
    sheet.addColumnAtCursor(newcol)
    sheet.calcRowIndex(newcol._rowindex)

@Sheet.api
def addcol_delta(sheet, vcolidx):
    if False:
        for i in range(10):
            print('nop')
    col = sheet.visibleCols[vcolidx]
    newcol = ColumnExpr('delta_' + col.name, type=col.type, _rowindex={}, expr='{0}-prev(row).{0}'.format(col.name))
    sheet.addcol_rowindex(newcol)
    return newcol

@Sheet.api
def addcol_rownum(sheet):
    if False:
        while True:
            i = 10
    newcol = Column('rownum', type=int, _rowindex={}, getter=lambda col, row: col._rowindex.get(col.sheet.rowid(row)))
    sheet.addcol_rowindex(newcol)
    return newcol
Sheet.addCommand(None, 'addcol-rownum', 'addcol_rownum()', helpstr='add column with original row ordering')
Sheet.addCommand(None, 'addcol-delta', 'addcol_delta(cursorVisibleColIndex)', helpstr='add column with delta of current column')