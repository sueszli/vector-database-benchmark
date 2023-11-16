def onValueChange(par, prev):
    if False:
        while True:
            i = 10
    return

def onValuesChanged(changes):
    if False:
        return 10
    for c in changes:
        par = c.par
        prev = c.prev
    return

def onPulse(par):
    if False:
        while True:
            i = 10
    if not parent.freeze.par.Benched.eval():
        if parent.freeze.par.Freeze.eval():
            op.Sound.SendFreeze('release', int(parent.track.par.Trackid.eval()))
            op('freezetimer').par.play = 0
            op('freezetimer').par.gotonextseg.pulse()
    return

def onExpressionChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onExportChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onEnableChange(par, val, prev):
    if False:
        print('Hello World!')
    return

def onModeChange(par, val, prev):
    if False:
        return 10
    return