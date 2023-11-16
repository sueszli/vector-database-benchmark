"""
# Usage

Duplicates are determined by the sheet's key columns.

If no key columns are specified, then a duplicate row is one where the values
of *all non-hidden* columns are exactly the same as a row that occurs earlier
in the sheet.

If key columns *are* specified, then duplicates are detected based on the
values in just those columns.

## Commands

- `select-duplicate-rows` sets the selection status in VisiData to `selected`
  for each row in the active sheet that is a duplicate of a prior row.

- `dedupe-rows` pushes a new sheet in which only non-duplicate rows in the
  active sheet are included.
"""
__author__ = 'Jeremy Singer-Vine <jsvine@gmail.com>'
from copy import copy
from visidata import Sheet, BaseSheet, asyncthread, Progress, vd

def gen_identify_duplicates(sheet):
    if False:
        while True:
            i = 10
    "\n    Takes a sheet, and returns a generator yielding a tuple for each row\n    encountered. The tuple's structure is `(row_object, is_dupe)`, where\n    is_dupe is True/False.\n\n    See note in Usage section above regarding how duplicates are determined.\n    "
    keyCols = sheet.keyCols
    cols_to_check = None
    if len(keyCols) == 0:
        vd.warning('No key cols specified. Using all columns.')
        cols_to_check = sheet.visibleCols
    else:
        cols_to_check = sheet.keyCols
    seen = set()
    for r in sheet.rows:
        vals = tuple((col.getValue(r) for col in cols_to_check))
        is_dupe = vals in seen
        if not is_dupe:
            seen.add(vals)
        yield (r, is_dupe)

@Sheet.api
@asyncthread
def select_duplicate_rows(sheet, duplicates=True):
    if False:
        return 10
    '\n    Given a sheet, sets the selection status in VisiData to `selected` for each\n    row that is a duplicate of a prior row.\n\n    If `duplicates = False`, then the behavior is reversed; sets the selection\n    status to `selected` for each row that is *not* a duplicate.\n    '
    before = len(sheet.selectedRows)
    gen = gen_identify_duplicates(sheet)
    prog = Progress(gen, gerund='selecting', total=sheet.nRows)
    for (row, is_dupe) in prog:
        if is_dupe == duplicates:
            sheet.selectRow(row)
    sel_count = len(sheet.selectedRows) - before
    more_str = ' more' if before > 0 else ''
    vd.status(f'selected {sel_count}{more_str} {sheet.rowtype}')

@Sheet.api
def dedupe_rows(sheet):
    if False:
        return 10
    '\n    Given a sheet, pushes a new sheet in which only non-duplicate rows are\n    included.\n    '
    vs = copy(sheet)
    vs.name += '_deduped'

    @asyncthread
    def _reload(self=vs):
        if False:
            while True:
                i = 10
        self.rows = []
        gen = gen_identify_duplicates(sheet)
        prog = Progress(gen, gerund='deduplicating', total=sheet.nRows)
        for (row, is_dupe) in prog:
            if not is_dupe:
                self.addRow(row)
    vs.reload = _reload
    return vs
BaseSheet.addCommand(None, 'select-duplicate-rows', 'sheet.select_duplicate_rows()', 'select each row that is a duplicate of a prior row')
BaseSheet.addCommand(None, 'dedupe-rows', 'vd.push(sheet.dedupe_rows())', 'open new sheet in which only non-duplicate rows in the active sheet are included')
vd.addMenuItems('\n    Row > Select > duplicate rows > select-duplicate-rows\n    Data > Deduplicate rows > dedupe-rows\n')
'\n# Changelog\n\n## 0.2.0 - 2021-09-22\n\nUse `vd.warning(...)` instead of `warning(...)`\n\n## 0.1.0 - 2020-10-09\n\nRevised for compatibility with VisiData 2.x\n\n## 0.0.1 - 2019-01-01\n\nInternal change, no external effects: Migrates from ._selectedRows to .selectedRows.\n\n## 0.0.0 - 2018-12-30\n\nInitial release.\n'