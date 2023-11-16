def isTypeMatch(defmodtype, modtype):
    if False:
        return 10
    tfilter = mod.vjzual.prepFilterList(defmodtype)
    if not tfilter:
        return True
    return modtype in tfilter

def getDefault(defaults, propname, modtype):
    if False:
        print('Hello World!')
    for i in range(1, defaults.numRows):
        if defaults[i, 'property'] != propname:
            continue
        if not isTypeMatch(defaults[i, 'type'].val, modtype):
            continue
        return defaults[i, 'value'].val
    return ''

def cook(dat):
    if False:
        for i in range(10):
            print('nop')
    dat.copy(dat.inputs[0])
    defs = dat.inputs[1]
    for c in defs.col(0)[1:]:
        if dat[0, c] is None:
            dat.appendCol(c)
    cols = [c.val for c in dat.row(0)[1:]]
    for mname in dat.col('name')[1:]:
        fmtvars = {'m': dat[mname, 'path'], 'mname': mname}
        mtype = dat[mname, 'type'].val
        for c in cols:
            cell = dat[mname, c]
            val = cell.val
            if val == '':
                val = getDefault(defs, c, mtype)
            if val == '-':
                cell.val = ''
            elif '{' in val:
                cell.val = val.format(**fmtvars)
            else:
                cell.val = val