import collections
import itertools
import functools
from copy import copy
from visidata import vd, VisiData, asyncthread, Sheet, Progress, IndexSheet, Column, CellColorizer, ColumnItem, SubColumnItem, TypedWrapper, ColumnsSheet, AttrDict
vd.help_join = '# Join Help\nHELPTODO'

@VisiData.api
def ensureLoaded(vd, sheets):
    if False:
        print('Hello World!')
    threads = [vs.ensureLoaded() for vs in sheets]
    threads = [t for t in threads if t]
    vd.status('loading %d sheets' % len(threads))
    return threads

@asyncthread
def _appendRowsAfterLoading(joinsheet, origsheets):
    if False:
        i = 10
        return i + 15
    if vd.ensureLoaded(origsheets):
        vd.sync()
    colnames = {c.name: c for c in joinsheet.visibleCols}
    for vs in origsheets:
        joinsheet.rows.extend(vs.rows)
        for c in vs.visibleCols:
            if c.name not in colnames:
                newcol = copy(c)
                colnames[c.name] = newcol
                joinsheet.addColumn(newcol)

@VisiData.api
def join_sheets_cols(vd, cols, jointype: str=''):
    if False:
        return 10
    'match joinkeys by cols in order per sheet.'
    sheetkeys = collections.defaultdict(list)
    for c in cols:
        sheetkeys[c.sheet].append(c)
    sheets = list(sheetkeys.keys())
    return JoinSheet('+'.join((vs.name for vs in sheets)), sources=sheets, sheetKeyCols=sheetkeys, jointype=jointype)

@Sheet.api
def openJoin(sheet, others, jointype=''):
    if False:
        i = 10
        return i + 15
    sheets = [sheet] + others
    sheets[1:] or vd.fail('join requires more than 1 sheet')
    if jointype == 'concat':
        name = '&'.join((vs.name for vs in sheets))
        sheettypes = set((type(vs) for vs in sheets))
        if len(sheettypes) != 1:
            vd.fail(f'only same sheet types can be concat-joined; use "append"')
        joinsheet = copy(sheet)
        joinsheet.name = name
        joinsheet.rows = []
        joinsheet.source = sheets
        _appendRowsAfterLoading(joinsheet, sheets)
        return joinsheet
    elif jointype == 'append':
        name = '&'.join((vs.name for vs in sheets))
        return ConcatSheet(name, source=sheets)
    nkeys = set((len(s.keyCols) for s in sheets))
    if 0 in nkeys or len(nkeys) != 1:
        vd.fail(f'all sheets must have the same number of key columns')
    if jointype == 'extend':
        vs = copy(sheets[0])
        vs.name = '+'.join((vs.name for vs in sheets))
        vs.sheetKeyCols = {vs: vs.keyCols for vs in sheets}
        vs.reload = functools.partial(ExtendedSheet_reload, vs, sheets)
        return vs
    else:
        return JoinSheet('+'.join((vs.name for vs in sheets)), sources=sheets, jointype=jointype, sheetKeyCols={s: s.keyCols for s in sheets})
vd.jointypes = [AttrDict(key=k, desc=v) for (k, v) in {'inner': 'only rows with matching keys on all sheets', 'outer': 'only rows with matching keys on first selected sheet', 'full': 'all rows from all sheets (union)', 'diff': 'only rows NOT in all sheets', 'append': 'all rows from all sheets; columns from all sheets', 'concat': 'all rows from all sheets; columns and type from first sheet', 'extend': 'only rows from first sheet; type from first sheet; columns from all sheets', 'merge': 'merge differences from other sheets into first sheet (including new rows)'}.items()]

def joinkey(sheetKeyCols, row):
    if False:
        print('Hello World!')
    return tuple((c.getDisplayValue(row) for c in sheetKeyCols))

def groupRowsByKey(sheets: dict, rowsBySheetKey, rowsByKey):
    if False:
        print('Hello World!')
    with Progress(gerund='grouping', total=sum((len(vs.rows) for vs in sheets)) * 2) as prog:
        for vs in sheets:
            rowsBySheetKey[vs] = collections.defaultdict(list)
            for r in vs.rows:
                prog.addProgress(1)
                key = joinkey(sheets[vs], r)
                rowsBySheetKey[vs][key].append(r)
        for vs in sheets:
            for r in vs.rows:
                prog.addProgress(1)
                key = joinkey(sheets[vs], r)
                if key not in rowsByKey:
                    rowsByKey[key] = [dict(crow) for crow in itertools.product(*[[(vs2, j) for j in rowsBySheetKey[vs2].get(key, [None])] for vs2 in sheets])]

class JoinKeyColumn(Column):

    def __init__(self, name='', keycols=None, **kwargs):
        if False:
            return 10
        super().__init__(name, type=keycols[0].type, width=keycols[0].width, **kwargs)
        self.keycols = keycols

    def calcValue(self, row):
        if False:
            while True:
                i = 10
        vals = set()
        for (i, c) in enumerate(self.keycols):
            if row[c.sheet] is not None:
                vals.add(c.getTypedValue(row[c.sheet]))
        if len(vals) != 1:
            vd.warning(f'inconsistent keys: ' + str(vals))
        return vals.pop()

    def putValue(self, row, value):
        if False:
            print('Hello World!')
        for (i, c) in enumerate(self.keycols):
            if row[c.sheet] is not None:
                c.setValues([row[c.sheet]], value)

    def recalc(self, sheet=None):
        if False:
            i = 10
            return i + 15
        Column.recalc(self, sheet)
        for c in self.keycols:
            c.recalc()

class MergeColumn(Column):

    def calcValue(self, row):
        if False:
            i = 10
            return i + 15
        for (vs, c) in reversed(list(self.cols.items())):
            if c:
                v = c.getTypedValue(row[vs])
                if v and (not isinstance(v, TypedWrapper)):
                    return v

    def putValue(self, row, value):
        if False:
            return 10
        for (vs, c) in reversed(list(self.cols.items())):
            c.setValue(row[vs], value)

    def isDiff(self, row, value):
        if False:
            for i in range(10):
                print('nop')
        col = list(self.cols.values())[0]
        return col and value != col.getValue(row[col.sheet])

class JoinSheet(Sheet):
    """Column-wise join/merge. `jointype` constructor arg should be one of jointypes."""
    colorizers = [CellColorizer(0, 'color_diff', lambda s, c, r, v: c and r and isinstance(c, MergeColumn) and c.isDiff(r, v.value))]
    sheetKeyCols = {}

    def loader(self):
        if False:
            print('Hello World!')
        sheets = self.sources
        vd.ensureLoaded(sheets)
        vd.sync()
        self.columns = []
        for (i, cols) in enumerate(itertools.zip_longest(*list(self.sheetKeyCols.values()))):
            self.addColumn(JoinKeyColumn(cols[0].name, keycols=cols))
        self.setKeys(self.columns)
        allcols = collections.defaultdict(dict)
        for (sheetnum, vs) in enumerate(sheets):
            for c in vs.visibleCols:
                if c not in self.sheetKeyCols[vs]:
                    allcols[c.name][vs] = c
        if self.jointype == 'merge':
            for (colname, cols) in allcols.items():
                self.addColumn(MergeColumn(colname, cols=cols))
        else:
            ctr = collections.Counter((c.name for vs in sheets for c in vs.visibleCols if c not in self.sheetKeyCols[vs]))
            for (sheetnum, vs) in enumerate(sheets):
                for c in vs.visibleCols:
                    if c not in self.sheetKeyCols[vs]:
                        newname = c.name if ctr[c.name] == 1 else '%s_%s' % (vs.name, c.name)
                        self.addColumn(SubColumnItem(vs, c, name=newname))
        rowsBySheetKey = {}
        rowsByKey = {}
        groupRowsByKey(self.sheetKeyCols, rowsBySheetKey, rowsByKey)
        self.rows = []
        with Progress(gerund='joining', total=len(rowsByKey)) as prog:
            for (k, combinedRows) in rowsByKey.items():
                prog.addProgress(1)
                if self.jointype in ['full', 'merge']:
                    for combinedRow in combinedRows:
                        self.addRow(combinedRow)
                elif self.jointype == 'inner':
                    for combinedRow in combinedRows:
                        if all((r is not None for r in combinedRow.values())):
                            self.addRow(combinedRow)
                elif self.jointype == 'outer':
                    for combinedRow in combinedRows:
                        if combinedRow[sheets[0]]:
                            self.addRow(combinedRow)
                elif self.jointype == 'diff':
                    for combinedRow in combinedRows:
                        if not all((r is not None for r in combinedRow.values())):
                            self.addRow(combinedRow)

class ExtendedColumn(Column):

    def calcValue(self, row):
        if False:
            while True:
                i = 10
        key = joinkey(self.firstJoinSource.keyCols, row)
        srcrow = self.rowsBySheetKey[self.srcsheet][key]
        if srcrow:
            return self.sourceCol.calcValue(srcrow[0])

    def putValue(self, row, value):
        if False:
            while True:
                i = 10
        key = joinkey(self.firstJoinSource.keyCols, row)
        srcrow = self.rowsBySheetKey[self.srcsheet][key]
        if len(srcrow) == 1:
            self.sourceCol.putValue(srcrow[0], value)
        else:
            vd.warning('failed to modify, not able to identify unique source row')

@asyncthread
def ExtendedSheet_reload(self, sheets):
    if False:
        while True:
            i = 10
    vd.ensureLoaded(sheets)
    vd.sync()
    self.columns = []
    for (i, c) in enumerate(sheets[0].keyCols):
        self.addColumn(copy(c))
    self.setKeys(self.columns)
    for (i, c) in enumerate(sheets[0].visibleCols):
        if c not in self.sheetKeyCols[c.sheet]:
            self.addColumn(copy(c))
    self.rowsBySheetKey = {}
    rowsByKey = {}
    for (sheetnum, vs) in enumerate(sheets[1:]):
        for c in vs.visibleCols:
            if c not in self.sheetKeyCols[c.sheet]:
                newname = '%s_%s' % (vs.name, c.name)
                newcol = ExtendedColumn(newname, srcsheet=vs, rowsBySheetKey=self.rowsBySheetKey, firstJoinSource=sheets[0], sourceCol=c)
                self.addColumn(newcol)
    groupRowsByKey(self.sheetKeyCols, self.rowsBySheetKey, rowsByKey)
    self.rows = []
    with Progress(gerund='joining', total=len(rowsByKey)) as prog:
        for (k, combinedRows) in rowsByKey.items():
            prog.addProgress(1)
            for combinedRow in combinedRows:
                if combinedRow[sheets[0]]:
                    self.addRow(combinedRow[sheets[0]])

class ConcatColumn(Column):
    """ConcatColumn(name, cols={srcsheet:srccol}, ...)"""

    def getColBySheet(self, s):
        if False:
            print('Hello World!')
        return self.cols.get(s, None)

    def calcValue(self, row):
        if False:
            return 10
        (srcSheet, srcRow) = row
        srcCol = self.getColBySheet(srcSheet)
        if srcCol:
            return srcCol.calcValue(srcRow)

    def setValue(self, row, v):
        if False:
            for i in range(10):
                print('nop')
        (srcSheet, srcRow) = row
        srcCol = self.getColBySheet(srcSheet)
        if srcCol:
            srcCol.setValue(srcRow, v)
        else:
            vd.fail('column not on source sheet')

class ConcatSheet(Sheet):
    """combination of multiple sheets by row concatenation. source=list of sheets. """
    columns = [ColumnItem('origin_sheet', 0, width=0)]

    def iterload(self):
        if False:
            i = 10
            return i + 15
        keyedcols = collections.defaultdict(dict)
        with Progress(gerund='joining', sheet=self, total=sum((vs.nRows for vs in self.source))) as prog:
            for sheet in self.source:
                if sheet.ensureLoaded():
                    vd.sync()
                for r in sheet.rows:
                    yield (sheet, r)
                    prog.addProgress(1)
                for (idx, col) in enumerate(sheet.visibleCols):
                    if not keyedcols[col.name]:
                        self.addColumn(ConcatColumn(col.name, cols=keyedcols[col.name], type=col.type))
                    if sheet in keyedcols[col.name]:
                        keyedcols[idx][sheet] = col
                        self.addColumn(ConcatColumn(col.name, cols=keyedcols[idx], type=col.type))
                    else:
                        keyedcols[col.name][sheet] = col

@VisiData.api
def chooseJointype(vd):
    if False:
        print('Hello World!')
    prompt = 'choose jointype: '

    def _fmt_aggr_summary(match, row, trigger_key):
        if False:
            for i in range(10):
                print('nop')
        formatted_jointype = match.formatted.get('key', row.key) if match else row.key
        r = ' ' * (len(prompt) - 3)
        r += f'[:keystrokes]{trigger_key}[/]  '
        r += formatted_jointype
        if row.desc:
            r += ' - '
            r += match.formatted.get('desc', row.desc) if match else row.desc
        return r
    return vd.activeSheet.inputPalette(prompt, vd.jointypes, value_key='key', formatter=_fmt_aggr_summary, help=vd.help_join, type='jointype')
IndexSheet.addCommand('&', 'join-selected', 'left, rights = someSelectedRows[0], someSelectedRows[1:]; vd.push(left.openJoin(rights, jointype=chooseJointype()))', 'merge selected sheets with visible columns from all, keeping rows according to jointype')
IndexSheet.bindkey('g&', 'join-selected')
Sheet.addCommand('&', 'join-sheets-top2', 'vd.push(openJoin(vd.sheets[1:2], jointype=chooseJointype()))', 'concatenate top two sheets in Sheets Stack')
Sheet.addCommand('g&', 'join-sheets-all', 'vd.push(openJoin(vd.sheets[1:], jointype=chooseJointype()))', 'concatenate all sheets in Sheets Stack')
ColumnsSheet.addCommand('&', 'join-sheets-cols', 'vd.push(join_sheets_cols(selectedRows, jointype=chooseJointype()))', '')
vd.addMenuItems('\n    Data > Join > selected sheets > join-selected\n    Data > Join > top two sheets > join-sheets-top2\n    Data > Join > all sheets > join-sheets-all\n')
IndexSheet.help += '\n    - `&` to join the selected sheets together\n'