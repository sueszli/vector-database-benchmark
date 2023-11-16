def onValueChange(par, prev):
    if False:
        i = 10
        return i + 15
    return

def onValuesChanged(changes):
    if False:
        i = 10
        return i + 15
    for c in changes:
        par = c.par
        prev = c.prev
    return

def onPulse(par):
    if False:
        for i in range(10):
            print('nop')
    if par.name == 'Reassign':
        op.Pharus.par.Reassign.pulse()
    elif par.name == 'Reset':
        op.Tracker.PassPulse(par.name)
    elif par.name == 'Performreset':
        op.Tracker.SetVal('Performer', 0)
    return

def onExpressionChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onExportChange(par, val, prev):
    if False:
        print('Hello World!')
    return

def onEnableChange(par, val, prev):
    if False:
        while True:
            i = 10
    return

def onModeChange(par, val, prev):
    if False:
        print('Hello World!')
    return