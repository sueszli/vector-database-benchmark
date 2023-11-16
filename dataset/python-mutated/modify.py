from copy import copy
from visidata import vd, VisiData, asyncthread
from visidata import Sheet, RowColorizer, CellColorizer, Column, BaseSheet, Progress
vd.theme_option('color_add_pending', 'green', 'color for rows pending add')
vd.theme_option('color_change_pending', 'reverse yellow', 'color for cells pending modification')
vd.theme_option('color_delete_pending', 'red', 'color for rows pending delete')
vd.option('overwrite', 'c', 'overwrite existing files {y=yes|c=confirm|n=no}')
vd.optalias('readonly', 'overwrite', 'n')
vd.optalias('ro', 'overwrite', 'n')
vd.optalias('y', 'overwrite', 'y')

@VisiData.api
def couldOverwrite(vd) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return True if overwrite might be allowed.'
    return vd.options.overwrite.startswith(('y', 'c'))

@VisiData.api
def confirmOverwrite(vd, path, msg: str=''):
    if False:
        while True:
            i = 10
    'Fail if file exists and overwrite not allowed.'
    if path.exists():
        msg = msg or f'{path.given} exists. overwrite? '
        ow = vd.options.overwrite
        if ow.startswith('c'):
            vd.confirm(msg)
        elif ow.startswith('y'):
            pass
        else:
            vd.fail('overwrite disabled')
    return True

@Sheet.lazy_property
def _deferredAdds(sheet):
    if False:
        while True:
            i = 10
    return dict()

@Sheet.lazy_property
def _deferredMods(sheet):
    if False:
        i = 10
        return i + 15
    return dict()

@Sheet.lazy_property
def _deferredDels(sheet):
    if False:
        print('Hello World!')
    return dict()
Sheet.colorizers += [RowColorizer(9, 'color_add_pending', lambda s, c, r, v: s.rowid(r) in s._deferredAdds), CellColorizer(8, 'color_change_pending', lambda s, c, r, v: s.isChanged(c, r)), RowColorizer(9, 'color_delete_pending', lambda s, c, r, v: s.isDeleted(r))]

@Sheet.api
def preloadHook(sheet):
    if False:
        while True:
            i = 10
    BaseSheet.preloadHook(sheet)
    sheet._deferredAdds.clear()
    sheet._deferredMods.clear()
    sheet._deferredDels.clear()

@Sheet.api
def rowAdded(self, row):
    if False:
        print('Hello World!')
    'Mark row as a deferred add-row'
    self._deferredAdds[self.rowid(row)] = row

    def _undoRowAdded(sheet, row):
        if False:
            return 10
        if sheet.rowid(row) not in sheet._deferredAdds:
            vd.warning('cannot undo to before commit')
            return
        del sheet._deferredAdds[sheet.rowid(row)]
    vd.addUndo(_undoRowAdded, self, row)

@Column.api
def cellChanged(col, row, val):
    if False:
        i = 10
        return i + 15
    'Mark cell at row for col as a deferred edit-cell'
    oldval = col.getValue(row)
    if oldval != val:
        rowid = col.sheet.rowid(row)
        if rowid not in col.sheet._deferredMods:
            rowmods = {}
            col.sheet._deferredMods[rowid] = (row, rowmods)
        else:
            (_, rowmods) = col.sheet._deferredMods[rowid]
        rowmods[col] = val

        def _undoCellChanged(col, row, oldval):
            if False:
                return 10
            if oldval == col.getSourceValue(row):
                if col.sheet.rowid(row) not in col.sheet._deferredMods:
                    vd.warning('cannot undo to before commit')
                    return
                del col.sheet._deferredMods[col.sheet.rowid(row)]
            else:
                (_, rowmods) = col.sheet._deferredMods[col.sheet.rowid(row)]
                rowmods[col] = oldval
        vd.addUndo(_undoCellChanged, col, row, oldval)

@Sheet.api
def rowDeleted(self, row):
    if False:
        while True:
            i = 10
    'Mark row as a deferred delete-row'
    self._deferredDels[self.rowid(row)] = row
    self.addUndoSelection()
    self.unselectRow(row)

    def _undoRowDeleted(sheet, row):
        if False:
            i = 10
            return i + 15
        if sheet.rowid(row) not in sheet._deferredDels:
            vd.warning('cannot undo to before commit')
            return
        del sheet._deferredDels[sheet.rowid(row)]
    vd.addUndo(_undoRowDeleted, self, row)

@Sheet.api
@asyncthread
def addRows(sheet, rows, index=None, undo=True):
    if False:
        return 10
    'Add *rows* after row at *index*.'
    addedRows = {}
    if index is None:
        index = len(sheet.rows)
    for (i, row) in enumerate(Progress(rows, gerund='adding')):
        addedRows[sheet.rowid(row)] = row
        sheet.addRow(row, index=index + i + 1)
        if sheet.defer:
            sheet.rowAdded(row)
    sheet.setModified()

    @asyncthread
    def _removeRows():
        if False:
            while True:
                i = 10
        sheet.deleteBy(lambda r, sheet=sheet, addedRows=addedRows: sheet.rowid(r) in addedRows, commit=True, undo=False)
    if undo:
        vd.addUndo(_removeRows)

@Sheet.api
def deleteBy(sheet, func, commit=False, undo=True):
    if False:
        while True:
            i = 10
    'Delete rows on sheet for which ``func(row)`` returns true.  Return number of rows deleted.\n    If sheet.defer is set and *commit* is True, remove rows immediately without deferring.\n    If undo is set to True, add an undo for deletion.'
    oldrows = copy(sheet.rows)
    oldidx = sheet.cursorRowIndex
    ndeleted = 0
    newCursorRow = None
    if sheet.defer and (not commit):
        ndeleted = 0
        for r in sheet.gatherBy(func, 'deleting'):
            sheet.rowDeleted(r)
            ndeleted += 1
        return ndeleted
    while oldidx < len(oldrows):
        if not func(oldrows[oldidx]):
            newCursorRow = sheet.rows[oldidx]
            break
        oldidx += 1
    sheet.rows.clear()
    for r in Progress(oldrows, 'deleting'):
        if not func(r):
            sheet.rows.append(r)
            if r is newCursorRow:
                sheet.cursorRowIndex = len(sheet.rows) - 1
        else:
            try:
                sheet.commitDeleteRow(r)
                ndeleted += 1
            except Exception as e:
                vd.exceptionCaught(e)
    if undo:
        vd.addUndo(setattr, sheet, 'rows', oldrows)
        sheet.setModified()
    if ndeleted:
        vd.status('deleted %s %s' % (ndeleted, sheet.rowtype))
    return ndeleted

@Sheet.api
def isDeleted(self, row):
    if False:
        while True:
            i = 10
    'Return True if *row* has been deferred for deletion.'
    return self.rowid(row) in self._deferredDels

@Sheet.api
def isChanged(self, col, row):
    if False:
        for i in range(10):
            print('nop')
    'Return True if cell at *row* for *col* has been deferred for modification.'
    try:
        (row, rowmods) = self._deferredMods[self.rowid(row)]
        newval = rowmods[col]
        curval = col.getSourceValue(row)
        return newval is None and curval is not None or (curval is None and newval is not None) or col.type(newval) != col.type(curval)
    except KeyError:
        return False
    except Exception:
        return False

@Column.api
def getSourceValue(col, row):
    if False:
        return 10
    'For deferred sheets, return value for *row* in this *col* as it would be in the source, without any deferred modifications applied.'
    return Column.calcValue(col, row)

@Sheet.api
def commitAdds(self):
    if False:
        while True:
            i = 10
    'Return the number of rows that have been marked for deferred add-row. Clear the marking.'
    nadded = 0
    nerrors = 0
    for row in self._deferredAdds.values():
        try:
            self.commitAddRow(row)
            nadded += 1
        except Exception as e:
            vd.exceptionCaught(e)
            nerrors += 1
    if nadded or nerrors:
        vd.status(f'added {nadded} {self.rowtype} ({nerrors} errors)')
    self._deferredAdds.clear()
    return nadded

@Sheet.api
def commitMods(sheet):
    if False:
        for i in range(10):
            print('nop')
    'Commit all deferred modifications (that are not from rows added or deleted in this commit.  Return number of cells changed.'
    (_, deferredmods, _) = sheet.getDeferredChanges()
    nmods = 0
    for (row, rowmods) in deferredmods.values():
        for (col, val) in rowmods.items():
            try:
                col.putValue(row, val)
                nmods += 1
            except Exception as e:
                vd.exceptionCaught(e)
    sheet._deferredMods.clear()
    return nmods

@Sheet.api
def commitDeletes(self):
    if False:
        return 10
    'Return the number of rows that have been marked for deletion. Delete the rows. Clear the marking.'
    ndeleted = self.deleteBy(self.isDeleted, commit=True, undo=False)
    if ndeleted:
        vd.status('deleted %s %s' % (ndeleted, self.rowtype))
    return ndeleted

@Sheet.api
def commitAddRow(self, row):
    if False:
        print('Hello World!')
    'To commit an added row.  Override per sheet type.'

@Sheet.api
def commitDeleteRow(self, row):
    if False:
        return 10
    'To commit a deleted row.  Override per sheet type.'

@asyncthread
@Sheet.api
def putChanges(sheet):
    if False:
        i = 10
        return i + 15
    'Commit changes to ``sheet.source``. May overwrite source completely without confirmation.  Overridable.'
    sheet.commitAdds()
    sheet.commitMods()
    sheet.commitDeletes()
    sheet._deferredDels.clear()

@Sheet.api
def getDeferredChanges(sheet):
    if False:
        print('Hello World!')
    'Return changes made to deferred sheets that have not been committed, as a tuple (added_rows, modified_rows, deleted_rows).  *modified_rows* does not include any *added_rows* or *deleted_rows*.\n\n        - *added_rows*: { rowid:row, ... }\n        - *modified_rows*: { rowid: (row, { col:val, ... }), ... }\n        - *deleted_rows*: { rowid: row }\n\n    *rowid* is from ``Sheet.rowid(row)``. *col* is an actual Column object.\n    '
    mods = {}
    for (row, rowmods) in sheet._deferredMods.values():
        rowid = sheet.rowid(row)
        if rowid not in sheet._deferredAdds and rowid not in sheet._deferredDels:
            mods[rowid] = (row, {col: val for (col, val) in rowmods.items() if sheet.isChanged(col, row)})
    return (sheet._deferredAdds, mods, sheet._deferredDels)

@Sheet.api
def changestr(self, adds, mods, deletes):
    if False:
        while True:
            i = 10
    'Return a str for status that outlines how many deferred changes are going to be committed.'
    cstr = ''
    if adds:
        cstr += 'add %d %s' % (len(adds), self.rowtype)
    if mods:
        if cstr:
            cstr += ' and '
        cstr += 'change %d values' % sum((len(rowmods) for (row, rowmods) in mods.values()))
    if deletes:
        if cstr:
            cstr += ' and '
        cstr += 'delete %d %s' % (len(deletes), self.rowtype)
    return cstr

@Sheet.api
def commit(sheet, *rows):
    if False:
        while True:
            i = 10
    'Commit all deferred changes on this sheet to original ``sheet.source``.'
    if not sheet.defer:
        vd.fail('commit-sheet is not enabled for this sheet type')
    (adds, mods, deletes) = sheet.getDeferredChanges()
    cstr = sheet.changestr(adds, mods, deletes)
    vd.confirmOverwrite(sheet.rootSheet().source, 'really ' + cstr + '? ')
    sheet.putChanges()
    sheet.hasBeenModified = False

@Sheet.api
def new_rows(sheet, n):
    if False:
        i = 10
        return i + 15
    return [sheet.newRow() for i in range(n)]
Sheet.addCommand('a', 'add-row', 'addRows([newRow()], index=cursorRowIndex); cursorDown(1)', 'append a blank row')
Sheet.addCommand('ga', 'add-rows', 'n=int(input("add rows: ", value=1)); addRows(new_rows(n), index=cursorRowIndex); cursorDown(1)', 'append N blank rows')
Sheet.addCommand('za', 'addcol-new', 'addColumnAtCursor(SettableColumn(input("column name: ")))', 'append an empty column')
Sheet.addCommand('gza', 'addcol-bulk', 'addColumnAtCursor(*(SettableColumn() for c in range(int(input("add columns: ")))))', 'append N empty columns')
Sheet.addCommand('z^S', 'commit-sheet', 'commit()', 'commit changes back to source.  not undoable!')
vd.addMenuItems('\n    File > Save > changes to source > commit-sheet\n    Row > Add > one row\n    Row > Add > multiple rows\n    Column > Add column > empty > one column > addcol-new\n    Column > Add column > empty > multiple columns > addcol-bulk\n')