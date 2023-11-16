def onOffToOn(channel, sampleIndex, val, prev):
    if False:
        i = 10
        return i + 15
    tid = int(parent.track.par.Trackid.eval())
    bench = parent.track.par.Benched.eval()
    if bench:
        op.Sound.SendBenched('break', tid)
        op('violationgrace').par.start.pulse()
    else:
        op.Sound.SendFreeze('break', tid)
        op('violationgrace').par.start.pulse()
    return

def whileOn(channel, sampleIndex, val, prev):
    if False:
        return 10
    tid = int(parent.track.par.Trackid.eval())
    bench = parent.track.par.Benched.eval()
    if bench:
        op.Sound.SendBenched('break', tid)
        op('violationgrace').par.start.pulse()
    else:
        op.Sound.SendFreeze('break', tid)
        op('violationgrace').par.start.pulse()
    return

def onOnToOff(channel, sampleIndex, val, prev):
    if False:
        print('Hello World!')
    return

def whileOff(channel, sampleIndex, val, prev):
    if False:
        return 10
    return

def onValueChange(channel, sampleIndex, val, prev):
    if False:
        print('Hello World!')
    return