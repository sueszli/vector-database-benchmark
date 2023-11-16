from copy import copy
from visidata import vd, asyncthread, Progress, Sheet, options, UNLOADED

@Sheet.api
def orderBy(sheet, *cols, reverse=False):
    if False:
        for i in range(10):
            print('nop')
    'Add *cols* to internal ordering and re-sort the rows accordingly.  Pass *reverse* as True to order these *cols* descending.  Pass empty *cols* (or cols[0] of None) to clear internal ordering.'
    if options.undo:
        vd.addUndo(setattr, sheet, '_ordering', copy(sheet._ordering))
        if sheet._ordering:
            vd.addUndo(sheet.sort)
        else:
            vd.addUndo(setattr, sheet, 'rows', copy(sheet.rows))
    do_sort = False
    if not cols or cols[0] is None:
        sheet._ordering.clear()
        cols = cols[1:]
        do_sort = True
    for c in cols:
        sheet._ordering.append((c, reverse))
        do_sort = True
    if do_sort:
        sheet.sort()

class Reversor:

    def __init__(self, obj):
        if False:
            i = 10
            return i + 15
        self.obj = obj

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return other.obj == self.obj

    def __lt__(self, other):
        if False:
            return 10
        return other.obj < self.obj

@Sheet.api
def sortkey(self, r, prog=None):
    if False:
        i = 10
        return i + 15
    ret = []
    for (col, reverse) in self._ordering:
        if isinstance(col, str):
            col = self.column(col)
        val = col.getTypedValue(r)
        ret.append(Reversor(val) if reverse else val)
    if prog:
        prog.addProgress(1)
    return ret

@Sheet.api
@asyncthread
def sort(self):
    if False:
        return 10
    'Sort rows according to the current internal ordering.'
    if self.rows is UNLOADED:
        return
    try:
        with Progress(gerund='sorting', total=self.nRows) as prog:
            self.rows.sort(key=lambda r, self=self, prog=prog: self.sortkey(r, prog=prog))
    except TypeError as e:
        vd.warning('sort incomplete due to TypeError; change column type')
        vd.exceptionCaught(e, status=False)
Sheet.addCommand('[', 'sort-asc', 'orderBy(None, cursorCol)', 'sort ascending by current column; replace any existing sort criteria')
Sheet.addCommand(']', 'sort-desc', 'orderBy(None, cursorCol, reverse=True)', 'sort descending by current column; replace any existing sort criteria ')
Sheet.addCommand('g[', 'sort-keys-asc', 'orderBy(None, *keyCols)', 'sort ascending by all key columns; replace any existing sort criteria')
Sheet.addCommand('g]', 'sort-keys-desc', 'orderBy(None, *keyCols, reverse=True)', 'sort descending by all key columns; replace any existing sort criteria')
Sheet.addCommand('z[', 'sort-asc-add', 'orderBy(cursorCol)', 'sort ascending by current column; add to existing sort criteria')
Sheet.addCommand('z]', 'sort-desc-add', 'orderBy(cursorCol, reverse=True)', 'sort descending by current column; add to existing sort criteria')
Sheet.addCommand('gz[', 'sort-keys-asc-add', 'orderBy(*keyCols)', 'sort ascending by all key columns; add to existing sort criteria')
Sheet.addCommand('gz]', 'sort-keys-desc-add', 'orderBy(*keyCols, reverse=True)', 'sort descending by all key columns; add to existing sort criteria')
vd.addMenuItems('\n    Column > Sort by > current column only > ascending > sort-asc\n    Column > Sort by > current column only > descending > sort-desc\n    Column > Sort by > current column also > ascending > sort-asc-add\n    Column > Sort by > current column also > descending > sort-desc-add\n    Column > Sort by > key columns > ascending > sort-keys-asc\n    Column > Sort by > key columns > descending > sort-keys-desc\n')