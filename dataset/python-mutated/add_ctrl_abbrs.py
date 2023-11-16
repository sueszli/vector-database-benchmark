def cook(dat):
    if False:
        i = 10
        return i + 15
    dat.clear()
    dat.copy(dat.inputs[0])
    dat.insertCol(['abbr'])
    for i in range(1, dat.numRows):
        name = dat[i, 'name'].val
        abbr = mod.vjzual.nameToAbbr(name)
        dat[i, 'abbr'] = abbr
    return