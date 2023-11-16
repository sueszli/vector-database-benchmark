def onValueChange(par, prev):
    if False:
        i = 10
        return i + 15
    return

def onValuesChanged(changes):
    if False:
        print('Hello World!')
    osc = parent.Guide.op('./oscout')
    gid = parent.mqguide.par.Guideid.eval() - 1
    table = op('cue_table')
    for c in changes:
        par = c.par
        prev = c.prev
        if par.name == 'Cue' and int(par.eval()) == 0:
            cue = str(prev)
            level = 0
            break
        elif par.name == 'Level':
            cue = str(parent.mqguide.par.Cue.eval())
            level = float(table[str(cue), 'Intensity'].val) / 100 * float(parent.mqguide.par.Level.eval())
        else:
            cue = par.eval()
            level = float(table[str(cue), 'Intensity'].val) / 100 * float(parent.mqguide.par.Level.eval())
            color = int(op('cue_table')[str(cue), 'Color'].val) + gid
            beam = int(op('cue_table')[str(cue), 'Beam'].val) + gid
            shutter = int(op('cue_table')[str(cue), 'Shutter'].val) + gid
            color_ex = f'/exec/14/{color}'
            beam_ex = f'/exec/12/{beam}'
            shut_ex = f'/exec/12/{shutter}'
            osc.sendOSC(color_ex, [int(100)], useNonStandardTypes=True)
            osc.sendOSC(beam_ex, [int(100)], useNonStandardTypes=True)
            osc.sendOSC(shut_ex, [int(100)], useNonStandardTypes=True)
        pass
    act = table[str(cue), 'Activation'].val
    act_ex = f'/exec/13/{int(act) + gid}'
    osc.sendOSC(act_ex, [float(level)], useNonStandardTypes=True)
    return

def onPulse(par):
    if False:
        print('Hello World!')
    return

def onExpressionChange(par, val, prev):
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    return