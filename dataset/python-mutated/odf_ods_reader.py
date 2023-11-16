from odf import opendocument
from odf.table import Table, TableRow, TableCell
from odf.text import P

class ODSReader(object):

    def __init__(self, file=None, content=None, clonespannedcolumns=None):
        if False:
            for i in range(10):
                print('nop')
        if not content:
            self.clonespannedcolumns = clonespannedcolumns
            self.doc = opendocument.load(file)
        else:
            self.clonespannedcolumns = clonespannedcolumns
            self.doc = content
        self.SHEETS = {}
        for sheet in self.doc.spreadsheet.getElementsByType(Table):
            self.readSheet(sheet)

    def readSheet(self, sheet):
        if False:
            return 10
        name = sheet.getAttribute('name')
        rows = sheet.getElementsByType(TableRow)
        arrRows = []
        for row in rows:
            arrCells = []
            cells = row.getElementsByType(TableCell)
            for (count, cell) in enumerate(cells, start=1):
                repeat = 0
                if count != len(cells):
                    repeat = cell.getAttribute('numbercolumnsrepeated')
                if not repeat:
                    repeat = 1
                    spanned = int(cell.getAttribute('numbercolumnsspanned') or 0)
                    if self.clonespannedcolumns is not None and spanned > 1:
                        repeat = spanned
                ps = cell.getElementsByType(P)
                textContent = u''
                for p in ps:
                    for n in p.childNodes:
                        if n.nodeType == 1 and n.tagName == 'text:span':
                            for c in n.childNodes:
                                if c.nodeType == 3:
                                    textContent = u'{}{}'.format(textContent, n.data)
                        if n.nodeType == 3:
                            textContent = u'{}{}'.format(textContent, n.data)
                if textContent:
                    if not textContent.startswith('#'):
                        for rr in range(int(repeat)):
                            arrCells.append(textContent)
                else:
                    for rr in range(int(repeat)):
                        arrCells.append('')
            if arrCells:
                arrRows.append(arrCells)
        self.SHEETS[name] = arrRows

    def getSheet(self, name):
        if False:
            i = 10
            return i + 15
        return self.SHEETS[name]

    def getFirstSheet(self):
        if False:
            return 10
        return next(iter(self.SHEETS.itervalues()))