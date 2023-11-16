import math
import os.path
from functools import singledispatch
from visidata import vd, Sheet, asyncthread, Progress, Column, VisiData, deduceType, anytype, getitemdef, ColumnsSheet

@Sheet.api
def getSampleRows(sheet):
    if False:
        return 10
    'Return list of sample rows centered around the cursor.'
    n = sheet.options.default_sample_size
    if n == 0 or n >= sheet.nRows:
        return sheet.rows
    vd.warning(f'sampling {n} rows')
    seq = sheet.rows
    start = math.ceil(sheet.cursorRowIndex - n / 2) % len(seq)
    end = (start + n) % len(seq)
    if start < end:
        return seq[start:end]
    return seq[start:] + seq[:end]

@Sheet.api
def expandCols(sheet, cols, rows=None, depth=0):
    if False:
        for i in range(10):
            print('nop')
    'expand all visible columns of containers to the given depth (0=fully)'
    ret = []
    if not rows:
        rows = sheet.getSampleRows()
    for col in cols:
        newcols = col.expand(rows)
        if depth != 1:
            ret.extend(sheet.expandCols(newcols, rows, depth - 1))
    return ret

@singledispatch
def _createExpandedColumns(sampleValue, col, rows):
    if False:
        print('Hello World!')
    'By default, a column is not expandable. Supported container types for\n    sampleValue trigger alternate, type-specific expansions.'
    return []

@_createExpandedColumns.register(dict)
def _(sampleValue, col, vals):
    if False:
        while True:
            i = 10
    'Build a set of columns to add, using the first occurrence of each key to\n    determine column type'
    newcols = {}
    for val in Progress(vals, 'expanding'):
        colsToAdd = set(val).difference(newcols)
        colsToAdd and newcols.update({k: deduceType(v) for (k, v) in val.items() if k in colsToAdd})
    return [ExpandedColumn(col.sheet.options.fmt_expand_dict % (col.name, k), type=v, origCol=col, expr=k) for (k, v) in newcols.items()]

def _createExpandedColumnsNamedTuple(col, val):
    if False:
        print('Hello World!')
    return [ExpandedColumn(col.sheet.options.fmt_expand_dict % (col.name, k), type=colType, origCol=col, expr=i) for (i, (k, colType)) in enumerate(zip(val._fields, (deduceType(v) for v in val)))]

@_createExpandedColumns.register(list)
@_createExpandedColumns.register(tuple)
def _(sampleValue, col, vals):
    if False:
        return 10
    'Use the longest sequence to determine the number of columns we need to\n    create, and their presumed types'

    def lenNoExceptions(v):
        if False:
            return 10
        try:
            return len(v)
        except Exception as e:
            return 0
    if hasattr(sampleValue, '_fields'):
        return _createExpandedColumnsNamedTuple(col, vals[0])
    longestSeq = max(vals, key=lenNoExceptions)
    colTypes = [deduceType(v) for v in longestSeq]
    return [ExpandedColumn(col.sheet.options.fmt_expand_list % (col.name, k), type=colType, origCol=col, expr=k) for (k, colType) in enumerate(colTypes)]

@Column.api
def expand(col, rows):
    if False:
        for i in range(10):
            print('nop')
    isNull = col.sheet.isNullFunc()
    nonNulls = [col.getTypedValue(row) for row in rows if not isNull(col.getValue(row))]
    if not nonNulls:
        return []
    expandedCols = _createExpandedColumns(nonNulls[0], col, nonNulls)
    idx = col.sheet.columns.index(col)
    for (i, c) in enumerate(expandedCols):
        col.sheet.addColumn(c, index=idx + i + 1)
    if expandedCols:
        col.hide()
    return expandedCols

@VisiData.api
class ExpandedColumn(Column):

    def calcValue(self, row):
        if False:
            return 10
        return getitemdef(self.origCol.getValue(row), self.expr)

    def setValue(self, row, value):
        if False:
            return 10
        self.origCol.getValue(row)[self.expr] = value

@Sheet.api
@asyncthread
def contract_cols(sheet, cols, depth=1):
    if False:
        print('Hello World!')
    'Remove any columns in cols with .origCol, and also remove others in sheet.columns which share those .origCol.  The inverse of expand.'
    vd.addUndo(setattr, sheet, 'columns', sheet.columns)
    for i in range(depth or 10000):
        colsToClose = [c for c in cols if getattr(c, 'origCol', None)]
        if not colsToClose:
            break
        origCols = set((c.origCol for c in colsToClose))
        for col in origCols:
            col.width = sheet.options.default_width
        sheet.columns = [col for col in sheet.columns if getattr(col, 'origCol', None) not in origCols]

@Sheet.api
@asyncthread
def expand_cols_deep(sheet, cols, rows=None, depth=0):
    if False:
        print('Hello World!')
    return sheet.expandCols(cols, rows=rows, depth=depth)

@ColumnsSheet.api
def contract_source_cols(sheet, cols):
    if False:
        print('Hello World!')
    prefix = os.path.commonprefix([c.name for c in cols])
    ret = ColumnGroup(prefix or 'group', prefix=prefix, sourceCols=cols)
    for c in cols:
        c.origCol = ret
    for vs in sheet.source:
        vd.addUndo(setattr, vs, 'columns', vs.columns)
        vs.columns[:] = [c for c in vs.columns if c not in cols]
    return ret

class ColumnGroup(Column):

    def calcValue(self, row):
        if False:
            while True:
                i = 10
        return {c.name[len(self.prefix):]: c.getValue(row) for c in self.sourceCols}

    def expand(self, rows):
        if False:
            while True:
                i = 10
        idx = self.sheet.columns.index(self)
        for (i, c) in enumerate(self.sourceCols):
            self.sheet.addColumn(c, index=idx + i + 1)
        self.hide()
        return self.sourceCols
Sheet.addCommand('(', 'expand-col', 'expand_cols_deep([cursorCol], depth=1)', 'expand current column of containers one level')
Sheet.addCommand('g(', 'expand-cols', 'expand_cols_deep(visibleCols, depth=1)', 'expand all visible columns of containers one level')
Sheet.addCommand('z(', 'expand-col-depth', 'expand_cols_deep([cursorCol], depth=int(input("expand depth=", value=0)))', 'expand current column of containers to given depth (0=fully)')
Sheet.addCommand('gz(', 'expand-cols-depth', 'expand_cols_deep(visibleCols, depth=int(input("expand depth=", value=0)))', 'expand all visible columns of containers to given depth (0=fully)')
Sheet.addCommand(')', 'contract-col', 'contract_cols([cursorCol])', 'remove current column and siblings from sheet columns and unhide parent')
Sheet.addCommand('g)', 'contract-cols', 'contract_cols(visibleCols)', 'remove all child columns and unhide toplevel parents')
Sheet.addCommand('z)', 'contract-col-depth', 'contract_cols([cursorCol], depth=int(input("contract depth=", value=0)))', 'remove current column and siblings from sheet columns and unhide parent')
Sheet.addCommand('gz)', 'contract-cols-depth', 'contract_cols(visibleCols, depth=int(input("contract depth=", value=0)))', 'remove all child columns and unhide toplevel parents')
ColumnsSheet.addCommand(')', 'contract-source-cols', 'source[0].addColumn(contract_source_cols(someSelectedRows), index=cursorRowIndex)', 'contract selected columns into column group')
vd.addMenuItems('\n    Column > Expand > one level > expand-col\n    Column > Expand > to depth N > expand-col-depth\n    Column > Expand > all columns one level > expand-cols\n    Column > Expand > all columns to depth > expand-cols-depth\n    Column > Contract > one level > contract-col\n    Column > Contract > N levels > contract-col-depth\n    Column > Contract > all columns one level > contract-cols\n    Column > Contract > all columns N levels > contract-cols-depth\n    Column > Contract > selected columns on source sheet > contract-source-cols\n')