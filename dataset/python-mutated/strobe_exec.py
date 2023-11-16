def onStart():
    if False:
        return 10
    return

def onCreate():
    if False:
        for i in range(10):
            print('nop')
    return

def onExit():
    if False:
        i = 10
        return i + 15
    return

def onFrameStart(frame):
    if False:
        print('Hello World!')
    strobechop = op('strobes')
    strobes = []
    numS = strobechop.numSamples
    for k in range(numS):
        tid = int(strobechop['Trackid'][k])
        act = strobechop['Freezeviolation'][k]
        tx = strobechop['Positionx'][k]
        ty = strobechop['Positiony'][k]
        if act:
            strobes.append([tid, tx, ty])
    parent.Sound.SendStrobes(strobes)
    return

def onFrameEnd(frame):
    if False:
        for i in range(10):
            print('nop')
    return

def onPlayStateChange(state):
    if False:
        for i in range(10):
            print('nop')
    return

def onDeviceChange():
    if False:
        i = 10
        return i + 15
    return

def onProjectPreSave():
    if False:
        while True:
            i = 10
    return

def onProjectPostSave():
    if False:
        return 10
    return