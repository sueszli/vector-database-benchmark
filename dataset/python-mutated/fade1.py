def LoadParamValue(module, tbl, pname):
    if False:
        i = 10
        return i + 15
    if not pname in ['xfade', 'blendop', 'swap']:
        return
    fullname = module.ModName + ':' + pname
    val = tbl[fullname, 1]
    if val is None or val == '':
        return
    if pname == 'xfade':
        op('blend').par.Xfade = float(val)
        return True
    elif pname == 'blendop':
        op('blend').par.Blendop = val
        return True
    elif pname == 'swap':
        op('blend').par.Swaporder = float(val) > 0.5
        return True

def SaveParamValue(module, tbl, pname):
    if False:
        for i in range(10):
            print('nop')
    if not pname in ['xfade', 'blendop', 'swap']:
        return
    fullname = module.ModName + ':' + pname
    val = None
    if pname == 'xfade':
        val = round(op('blend').par.Xfade.val, 4)
    elif pname == 'blendop':
        val = op('blend').par.Blendop.val
    elif pname == 'swap':
        val = 1 if op('blend').par.Swaporder.val else 0
    if val is None:
        return
    mod.vjzual.updateTableRow(tbl, fullname, {'value': val}, addMissing=True)
    return True