from typing import Mapping
import inspect
import math
from visidata import vd, asyncthread, ENTER, deduceType
from visidata import Sheet, Column, VisiData, ColumnItem, TableSheet, BaseSheet, Progress, ColumnAttr, SuspendCurses, TextSheet
import visidata
vd.option('visibility', 0, 'visibility level')
vd.option('default_sample_size', 100, 'number of rows to sample for regex.split (0=all)', replay=True)
vd.option('fmt_expand_dict', '%s.%s', 'format str to use for names of columns expanded from dict (colname, key)')
vd.option('fmt_expand_list', '%s[%s]', 'format str to use for names of columns expanded from list (colname, index)')

class PythonSheet(Sheet):

    def openRow(self, row):
        if False:
            return 10
        return PyobjSheet('%s[%s]' % (self.name, self.keystr(row)), source=row)

class InferColumnsSheet(Sheet):
    _rowtype = dict

    def resetCols(self):
        if False:
            i = 10
            return i + 15
        self._knownKeys = set()
        super().resetCols()

    def addColumn(self, *cols, index=None):
        if False:
            return 10
        for c in cols:
            self._knownKeys.add(c.expr or c.name)
        return super().addColumn(*cols, index=index)

    def addRow(self, row, index=None):
        if False:
            i = 10
            return i + 15
        ret = super().addRow(row, index=index)
        for k in row:
            if k not in self._knownKeys:
                self.addColumn(ColumnItem(k, type=deduceType(row[k])))
        return ret
InferColumnsSheet.init('_knownKeys', set, copy=True)
InferColumnsSheet.init('_ordering', list, copy=True)

@VisiData.global_api
def view(vd, obj):
    if False:
        for i in range(10):
            print('nop')
    vd.run(PyobjSheet(getattr(obj, '__name__', ''), source=obj))

def getPublicAttrs(obj):
    if False:
        return 10
    'Return all public attributes (not methods or `_`-prefixed) on object.'
    return [k for k in dir(obj) if not k.startswith('_') and (not callable(getattr(obj, k)))]

def PyobjColumns(obj):
    if False:
        while True:
            i = 10
    'Return columns for each public attribute on an object.'
    return [ColumnAttr(k, type=deduceType(getattr(obj, k))) for k in getPublicAttrs(obj)]

def AttrColumns(attrnames):
    if False:
        while True:
            i = 10
    'Return column names for all elements of list `attrnames`.'
    return [ColumnAttr(name) for name in attrnames]

def SheetList(*names, **kwargs):
    if False:
        return 10
    'Creates a Sheet from a list of homogeneous dicts or namedtuples.'
    src = kwargs.get('source', None)
    if not src:
        vd.warning('no content in %s' % names)
        return Sheet(*names, **kwargs)
    if isinstance(src[0], Mapping):
        return ListOfDictSheet(*names, **kwargs)
    elif isinstance(src[0], tuple):
        if getattr(src[0], '_fields', None):
            return ListOfNamedTupleSheet(*names, **kwargs)
    return ListOfPyobjSheet(*names, **kwargs)

class ListOfPyobjSheet(PythonSheet):
    rowtype = 'python objects'

    def loader(self):
        if False:
            while True:
                i = 10
        self.rows = self.source
        self.columns = []
        self.addColumn(Column(self.name, getter=lambda col, row: row, setter=lambda col, row, val: setitem(col.sheet.source, col.sheet.source.index(row), val)))
        for c in PyobjColumns(self.rows[0]):
            self.addColumn(c)
        if len(self.columns) > 1:
            self.columns[0].width = 0

class ListOfDictSheet(PythonSheet):
    rowtype = 'dicts'

    def reload(self):
        if False:
            i = 10
            return i + 15
        self.columns = []
        self._knownKeys = set()
        for row in self.source:
            for k in row:
                if k not in self._knownKeys:
                    self.addColumn(ColumnItem(k, k, type=deduceType(row[k])))
                    self._knownKeys.add(k)
        self.rows = self.source

class ListOfNamedTupleSheet(PythonSheet):
    rowtype = 'namedtuples'

    def reload(self):
        if False:
            print('Hello World!')
        self.columns = []
        for (i, k) in enumerate(self.source[0]._fields):
            self.addColumn(ColumnItem(k, i))
        self.rows = self.source

class SheetNamedTuple(PythonSheet):
    """a single namedtuple, with key and value columns"""
    rowtype = 'values'
    columns = [ColumnItem('name', 0), ColumnItem('value', 1)]

    def __init__(self, *names, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*names, **kwargs)

    def reload(self):
        if False:
            i = 10
            return i + 15
        self.rows = list(zip(self.source._fields, self.source))

    def openRow(self, row):
        if False:
            print('Hello World!')
        return PyobjSheet(f'{self.name}.{row[0]}', source=row[1])

class SheetDict(PythonSheet):
    rowtype = 'items'
    columns = [Column('key'), Column('value', getter=lambda c, r: c.sheet.source[r], setter=lambda c, r, v: setitem(c.sheet.source, r, v))]
    nKeys = 1

    def reload(self):
        if False:
            while True:
                i = 10
        self.rows = list(self.source.keys())

    def openRow(self, row):
        if False:
            print('Hello World!')
        return PyobjSheet(f'{self.name}.{row}', source=self.source[row])

class ColumnSourceAttr(Column):
    """Use row as attribute name on sheet source"""

    def calcValue(self, attrname):
        if False:
            i = 10
            return i + 15
        return getattr(self.sheet.source, attrname)

    def setValue(self, attrname, value):
        if False:
            return 10
        return setattr(self.sheet.source, attrname, value)

def docstring(obj, attr):
    if False:
        print('Hello World!')
    v = getattr(obj, attr)
    if callable(v):
        return v.__doc__
    return '<type %s>' % type(v).__name__

class PyobjSheet(PythonSheet):
    """Generic Sheet for any Python object.  Return specialized subclasses for lists of objects, namedtuples, and dicts."""
    rowtype = 'attributes'
    columns = [Column('attribute'), ColumnSourceAttr('value'), Column('signature', width=0, getter=lambda c, r: dict(inspect.signature(getattr(c.sheet.source, r)).parameters)), Column('docstring', getter=lambda c, r: docstring(c.sheet.source, r))]
    nKeys = 1

    def __new__(cls, *names, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return Sheet object of appropriate type for given sources in `args`.'
        pyobj = kwargs.get('source', object())
        if isinstance(pyobj, list) or isinstance(pyobj, tuple):
            if getattr(pyobj, '_fields', None):
                return SheetNamedTuple(*names, **kwargs)
            else:
                return SheetList(*names, **kwargs)
        elif isinstance(pyobj, Mapping):
            return SheetDict(*names, **kwargs)
        elif isinstance(pyobj, str):
            return TextSheet(*names, source=pyobj.splitlines())
        elif isinstance(pyobj, bytes):
            return TextSheet(*names, source=pyobj.decode(options.encoding).splitlines())
        elif isinstance(pyobj, object):
            obj = super().__new__(cls)
            return obj
        else:
            vd.error("cannot load '%s' as pyobj" % type(pyobj).__name__)

    def reload(self):
        if False:
            while True:
                i = 10
        self.rows = []
        vislevel = self.options.visibility
        for r in dir(self.source):
            try:
                if vislevel <= 2 and r.startswith('__'):
                    continue
                if vislevel <= 1 and r.startswith('_'):
                    continue
                if vislevel <= 0 and callable(getattr(self.source, r)):
                    continue
            except Exception:
                pass
            self.addRow(r)

    def openRow(self, row):
        if False:
            print('Hello World!')
        'dive further into Python object'
        v = getattr(self.source, row)
        return PyobjSheet(self.name + '.' + str(row), source=v() if callable(v) else v)

@TableSheet.api
def openRow(sheet, row, rowidx=None):
    if False:
        return 10
    'Return Sheet diving into *row*.'
    if rowidx is None:
        k = sheet.keystr(row) or str(sheet.cursorRowIndex)
    else:
        k = rowidx
    name = f'{sheet.name}[{k}]'
    return TableSheet(name, rows=sheet.visibleCols, sourceRow=sheet.cursorRow, columns=[Column('column', getter=lambda c, r: r.name), Column('value', getter=lambda c, r: r.getTypedValue(c.sheet.sourceRow), setter=lambda c, r, v: r.setValue(c.sheet.sourceRow, v))], nKeys=1)

@TableSheet.api
def openCell(sheet, col, row, rowidx=None):
    if False:
        for i in range(10):
            print('nop')
    'Return Sheet diving into cell at *row* in *col*.'
    if rowidx is None:
        k = sheet.keystr(row) or str(sheet.cursorRowIndex)
    else:
        k = rowidx
    name = f'{sheet.name}[{k}].{col.name}'
    return PyobjSheet(name, source=col.getTypedValue(row))

@TableSheet.api
def openRowPyobj(sheet, rowidx):
    if False:
        i = 10
        return i + 15
    'Return Sheet of raw Python object of row.'
    return PyobjSheet('%s[%s]' % (sheet.name, rowidx), source=sheet.rows[rowidx])

@TableSheet.api
def openCellPyobj(sheet, col, rowidx):
    if False:
        for i in range(10):
            print('nop')
    'Return Sheet of raw Python object of cell.'
    name = f'{sheet.name}[{rowidx}].{col.name}'
    return PyobjSheet(name, source=col.getValue(sheet.rows[rowidx]))

@BaseSheet.api
def pyobj_expr(sheet):
    if False:
        for i in range(10):
            print('nop')

    def launch_repl(v, i):
        if False:
            return 10
        import code
        with SuspendCurses():
            code.InteractiveConsole(locals=locals()).interact()
        return (v, i)
    expr = vd.input('eval: ', 'expr', completer=visidata.CompleteExpr(), bindings={'^X': launch_repl})
    vd.push(PyobjSheet(expr, source=sheet.evalExpr(expr)))
BaseSheet.addCommand('^X', 'pyobj-expr', 'pyobj_expr()', 'evaluate Python expression and open result as Python object')
BaseSheet.addCommand('', 'exec-python', 'expr = input("exec: ", "expr", completer=CompleteExpr()); exec(expr, getGlobals(), LazyChainMap(sheet, *vd.contexts, locals=vd.getGlobals()))', 'execute Python statement with expression scope')
BaseSheet.addCommand('g^X', 'import-python', 'modname=input("import: ", type="import_python"); exec("import "+modname, getGlobals())', 'import Python module in the global scope')
BaseSheet.addCommand('z^X', 'pyobj-expr-row', 'expr = input("eval over current row: ", "expr", completer=CompleteExpr()); vd.push(PyobjSheet(expr, source=evalExpr(expr, row=cursorRow)))', 'evaluate Python expression, in context of current row, and open result as Python object')
Sheet.addCommand('^Y', 'pyobj-row', 'status(type(cursorRow).__name__); vd.push(openRowPyobj(cursorRowIndex))', 'open current row as Python object')
Sheet.addCommand('z^Y', 'pyobj-cell', 'status(type(cursorValue).__name__); vd.push(openCellPyobj(cursorCol, cursorRowIndex))', 'open current cell as Python object')
BaseSheet.addCommand('g^Y', 'pyobj-sheet', 'status(type(sheet).__name__); vd.push(PyobjSheet(sheet.name+"_sheet", source=sheet))', 'open current sheet as Python object')
Sheet.addCommand('', 'open-row-basic', 'vd.push(TableSheet.openRow(sheet, cursorRow))', 'dive into current row as basic table (ignoring subsheet dive)')
Sheet.addCommand(ENTER, 'open-row', 'vd.push(openRow(cursorRow))', 'open current row with sheet-specific dive')
Sheet.addCommand('z' + ENTER, 'open-cell', 'vd.push(openCell(cursorCol, cursorRow))', 'open sheet with copies of rows referenced in current cell')
Sheet.addCommand('g' + ENTER, 'dive-selected', 'for r in selectedRows: vd.push(openRow(r))', 'open sheet with copies of rows referenced in selected rows')
Sheet.addCommand('gz' + ENTER, 'dive-selected-cells', 'for r in selectedRows: vd.push(openCell(cursorCol, r))', 'open sheet with copies of rows referenced in selected rows')
PyobjSheet.addCommand('v', 'visibility', 'sheet.options.visibility = 0 if sheet.options.visibility else 2; reload()', 'toggle show/hide for methods and hidden properties')
PyobjSheet.addCommand('gv', 'show-hidden', 'sheet.options.visibility = 2; reload()', 'show methods and hidden properties')
PyobjSheet.addCommand('zv', 'hide-hidden', 'sheet.options.visibility -= 1; reload()', 'hide methods and hidden properties')
vd.addGlobals({'PythonSheet': PythonSheet, 'ListOfDictSheet': ListOfDictSheet, 'SheetDict': SheetDict, 'InferColumnsSheet': InferColumnsSheet, 'PyobjSheet': PyobjSheet, 'view': view})
vd.addMenuItems('\n    View > Visibility > Methods and dunder attributes > show > show-hidden\n    View > Visibility > Methods and dunder attributes > hide > hide-hidden\n    Row > Dive into > open-row\n    System > Python > import library > import-python\n    System > Python > current sheet > pyobj-sheet\n    System > Python > current row > pyobj-row\n    System > Python > current cell > pyobj-cell\n    System > Python > expression > pyobj-expr\n    System > Python > exec() > exec-python\n')