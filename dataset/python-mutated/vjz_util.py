__author__ = 'tekt'

def argToOp(arg):
    if False:
        print('Hello World!')
    if not arg:
        return None
    if isinstance(arg, str):
        o = op(arg)
        if not o:
            raise Exception('operator not found: ' + arg)
        return o
    return arg

def argToPath(arg):
    if False:
        return 10
    if not arg:
        return ''
    if isinstance(arg, str):
        return arg
    if hasattr(arg, 'path'):
        return arg.path
    return arg

def updateTableRow(tbl, rowKey, vals, addMissing=False, ignoreMissingCols=False):
    if False:
        i = 10
        return i + 15
    tbl = argToOp(tbl)
    if not tbl:
        return
    if not tbl[rowKey, 0]:
        if not addMissing:
            raise Exception('row ' + rowKey + ' not found in table ' + tbl)
        else:
            tbl.appendRow([rowKey])
    for colKey in vals:
        v = vals[colKey]
        if ignoreMissingCols and tbl[rowKey, colKey] is None:
            continue
        tbl[rowKey, colKey] = v if v is not None else ''

def overrideRows(tbl, overrides):
    if False:
        for i in range(10):
            print('nop')
    tbl = argToOp(tbl)
    if not tbl:
        return
    for key in overrides:
        tbl[key, 1] = overrides[key]