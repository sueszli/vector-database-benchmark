def onStart():
    if False:
        return 10
    return

def onCreate():
    if False:
        i = 10
        return i + 15
    return

def onExit():
    if False:
        i = 10
        return i + 15
    return

def onFrameStart(frame):
    if False:
        return 10
    zapchop = op('zaps')
    zaps = []
    numZ = zapchop.numSamples
    for k in range(numZ):
        tid = int(zapchop['Trackid'][k])
        tx = zapchop['Positionx'][k]
        ty = zapchop['Positiony'][k]
        zaps.append([tid, tx, ty])
    parent.Sound.SendZaps(zaps)
    return

def onFrameEnd(frame):
    if False:
        return 10
    return

def onPlayStateChange(state):
    if False:
        i = 10
        return i + 15
    return

def onDeviceChange():
    if False:
        print('Hello World!')
    return

def onProjectPreSave():
    if False:
        print('Hello World!')
    return

def onProjectPostSave():
    if False:
        for i in range(10):
            print('nop')
    return