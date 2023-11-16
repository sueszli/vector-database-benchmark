def onStart():
    if False:
        return 10
    return

def onCreate():
    if False:
        return 10
    return

def onExit():
    if False:
        while True:
            i = 10
    return

def onFrameStart(frame):
    if False:
        while True:
            i = 10
    return

def onFrameEnd(frame):
    if False:
        i = 10
        return i + 15
    synth = op('synth_set_dat')
    args = []
    for i in range(1, 51):
        pitch = int(synth[i, 'Trackid'].val)
        args.append(pitch)
        level = float(synth[i, 'Level'].val or 0)
        args.append(level)
        posx = float(synth[i, 'Positionx'].val or 0)
        args.append(posx)
        posy = float(synth[i, 'Positiony'].val or 0)
        args.append(posy)
    op.Sound.SendSynth(f'/synth', args)
    return

def onPlayStateChange(state):
    if False:
        return 10
    return

def onDeviceChange():
    if False:
        return 10
    return

def onProjectPreSave():
    if False:
        while True:
            i = 10
    return

def onProjectPostSave():
    if False:
        for i in range(10):
            print('nop')
    return