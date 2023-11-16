def onValueChange(par, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def onValuesChanged(changes):
    if False:
        for i in range(10):
            print('nop')
    for c in changes:
        par = c.par
        prev = c.prev
    return

def onPulse(par):
    if False:
        return 10
    parent.Tracker.PassPulse(par.name)
    return

def onExpressionChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onExportChange(par, val, prev):
    if False:
        while True:
            i = 10
    return

def onEnableChange(par, val, prev):
    if False:
        print('Hello World!')
    return

def onModeChange(par, val, prev):
    if False:
        return 10
    return