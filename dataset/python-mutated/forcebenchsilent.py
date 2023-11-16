def onValueChange(par, prev):
    if False:
        return 10
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
        while True:
            i = 10
    if not parent.freeze.par.Grace.eval() and (not parent.freeze.par.Benched.eval()):
        ft = op('freezetimer')
        ft.par.gotodone.pulse()
        ft.par.play = 1
        run(op('script_freezesound')[0, 0].val, delayFrames=2)
        run(op('script_freezeposition')[0, 0].val, delayFrames=2)
        op('violationgrace').par.start.pulse()
    return

def onExpressionChange(par, val, prev):
    if False:
        print('Hello World!')
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
        i = 10
        return i + 15
    return