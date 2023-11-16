from visidata import vd, VisiData, BaseSheet, Sheet, Column, AttrColumn, ItemColumn, setitem, asyncthread

class IndexSheet(Sheet):
    """Base class for tabular sheets with rows that are Sheets."""
    help = '\n    # Index Sheet\n\n    This is a list of sheets from `{sheet.source}`.\n\n    - `Enter` to open {sheet.cursorRow}\n    - `g Enter` to open all selected sheets\n    '
    rowtype = 'sheets'
    columns = [Column('name', getter=lambda c, r: r.names[-1], setter=lambda c, r, v: setitem(r.names, -1, v)), AttrColumn('rows', 'nRows', type=int, width=9), AttrColumn('cols', 'nCols', type=int), AttrColumn('keys', 'keyColNames'), AttrColumn('source')]
    nKeys = 1

    def newRow(self):
        if False:
            i = 10
            return i + 15
        return Sheet('', columns=[ItemColumn('', 0)], rows=[])

    def openRow(self, row):
        if False:
            print('Hello World!')
        return row

    def getSheet(self, k):
        if False:
            print('Hello World!')
        for vs in self.rows:
            if vs.name == k:
                return vs

    def addRow(self, sheet, **kwargs):
        if False:
            i = 10
            return i + 15
        super().addRow(sheet, **kwargs)
        if not self.options.load_lazy and (not sheet.options.load_lazy):
            sheet.ensureLoaded()

    @asyncthread
    def reloadSheets(self, sheets):
        if False:
            print('Hello World!')
        for vs in vd.Progress(sheets):
            vs.reload()

class SheetsSheet(IndexSheet):
    columns = [AttrColumn('name'), AttrColumn('type', '__class__.__name__'), AttrColumn('pane', type=int), Column('shortcut', getter=lambda c, r: getattr(r, 'shortcut'), setter=lambda c, r, v: setattr(r, '_shortcut', v)), AttrColumn('nRows', type=int), AttrColumn('nCols', type=int), AttrColumn('nVisibleCols', type=int), AttrColumn('cursorDisplay'), AttrColumn('keyColNames'), AttrColumn('source'), AttrColumn('progressPct')]
    precious = False
    nKeys = 1

    def reload(self):
        if False:
            i = 10
            return i + 15
        self.rows = self.source

    def sort(self):
        if False:
            print('Hello World!')
        self.rows[1:] = sorted(self.rows[1:], key=self.sortkey)

class GlobalSheetsSheet(SheetsSheet):

    def sort(self):
        if False:
            while True:
                i = 10
        IndexSheet.sort(self)

@VisiData.lazy_property
def sheetsSheet(vd):
    if False:
        print('Hello World!')
    return SheetsSheet('sheets', source=vd.sheets)

@VisiData.lazy_property
def allSheetsSheet(vd):
    if False:
        while True:
            i = 10
    return GlobalSheetsSheet('sheets_all', source=vd.allSheets)

@Sheet.api
def nextRow(sheet, n=1):
    if False:
        i = 10
        return i + 15
    sheet.cursorRowIndex += n
    sheet.checkCursor()
    return sheet.rows[sheet.cursorRowIndex]
vd.addCommand('S', 'sheets-stack', 'vd.push(vd.sheetsSheet)', 'open Sheets Stack: join or jump between the active sheets on the current stack')
vd.addCommand('gS', 'sheets-all', 'vd.push(vd.allSheetsSheet)', 'open Sheets Sheet: join or jump between all sheets from current session')
BaseSheet.addCommand('g>', 'open-source-next', 'vd.replace(openSource(source.nextRow())) if isinstance(source, IndexSheet) else fail("parent sheet must be Index Sheet")', 'open next sheet on parent index sheet')
BaseSheet.addCommand('g<', 'open-source-prev', 'vd.replace(openSource(source.nextRow(-1))) if isinstance(source, IndexSheet) else fail("parent sheet must be Index Sheet")', 'open prev sheet on parent index sheet')
IndexSheet.addCommand('g^R', 'reload-selected', 'reloadSheets(selectedRows or rows)', 'reload all selected sheets')
SheetsSheet.addCommand('gC', 'columns-selected', 'vd.push(ColumnsSheet("all_columns", source=selectedRows))', 'open Columns Sheet with all visible columns from selected sheets')
SheetsSheet.addCommand('z^C', 'cancel-row', 'cancelThread(*cursorRow.currentThreads)', 'abort async thread for current sheet')
SheetsSheet.addCommand('gz^C', 'cancel-rows', 'for vs in selectedRows: cancelThread(*vs.currentThreads)', 'abort async threads for selected sheets')
SheetsSheet.addCommand('Enter', 'open-row', 'dest=cursorRow; vd.sheets.remove(sheet) if not sheet.precious else None; vd.push(openRow(dest))', 'open sheet referenced in current row')
vd.addGlobals(IndexSheet=IndexSheet, SheetsSheet=SheetsSheet, GlobalSheetsSheet=GlobalSheetsSheet)