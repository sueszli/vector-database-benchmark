def LoadParamValue(module, tbl, pname):
    if False:
        print('Hello World!')
    if not pname in []:
        return
    fullname = module.ModName + ':' + pname
    val = tbl[fullname, 1]
    if val is None or val == '':
        return

def SaveParamValue(module, tbl, pname):
    if False:
        for i in range(10):
            print('nop')
    if not pname in []:
        return
    fullname = module.ModName + ':' + pname
    val = None
    if val is None:
        return
    mod.vjzual.updateTableRow(tbl, fullname, {'value': val}, addMissing=True)
    return True