def onValueChange(par, prev):
    if False:
        i = 10
        return i + 15
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
        for i in range(10):
            print('nop')
    if par.name == 'Initround':
        parent().InitRound()
    elif par.name == 'Startround':
        parent().StartRound()
    elif par.name == 'Pauseround':
        parent().PauseRound()
    elif par.name == 'Resumeround':
        parent().ResumeRound()
    elif par.name == 'Stopround':
        parent().StopRound()
    elif par.name == 'Evaluateround':
        parent().EvaluateRound()
    return

def onExpressionChange(par, val, prev):
    if False:
        print('Hello World!')
    return

def onExportChange(par, val, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def onEnableChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onModeChange(par, val, prev):
    if False:
        i = 10
        return i + 15
    return