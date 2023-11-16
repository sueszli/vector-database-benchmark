def onOffToOn(channel, sampleIndex, val, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def whileOn(channel, sampleIndex, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onOnToOff(channel, sampleIndex, val, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def whileOff(channel, sampleIndex, val, prev):
    if False:
        i = 10
        return i + 15
    return

def onValueChange(channel, sampleIndex, val, prev):
    if False:
        print('Hello World!')
    fc = op('freeze_chans')
    freeze = fc['Freeze']
    bench = fc['Benched']
    tid = parent.track.par.Trackid
    score = parent.track.par.Score
    if channel.name == 'Freeze':
        if not bench:
            if val == 0:
                op.Sound.SendFreeze('release', tid, score)
            elif val == 1:
                op.Sound.SendFreeze('start', tid, score)
    elif bench == 0:
        op.Sound.SendBenched('release', tid, score)
    elif bench == 1:
        op.Sound.SendBenched('start', tid, score)
    return