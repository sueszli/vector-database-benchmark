def onValueChange(par, prev):
    if False:
        i = 10
        return i + 15
    return

def onValuesChanged(changes):
    if False:
        for i in range(10):
            print('nop')
    return

def onPulse(par):
    if False:
        for i in range(10):
            print('nop')
    rt = me.parent.Roundtimer
    name = par.name
    if name == 'Reinit':
        rt.ReInit()
    elif name == 'Gointro':
        rt.GoIntro()
    elif name == 'Endintro':
        rt.EndIntro()
    elif name == 'Goround':
        rt.GoRound()
    elif name == 'Endround':
        rt.EndRound()
    elif name == 'Gooutro':
        rt.GoOutro()
    elif name == 'Endoutro':
        rt.EndOutro()
    return

def onExpressionChange(par, val, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def onExportChange(par, val, prev):
    if False:
        while True:
            i = 10
    return

def onEnableChange(par, val, prev):
    if False:
        return 10
    return

def onModeChange(par, val, prev):
    if False:
        return 10
    return