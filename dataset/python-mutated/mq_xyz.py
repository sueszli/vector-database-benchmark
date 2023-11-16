def onStart():
    if False:
        for i in range(10):
            print('nop')
    return

def onCreate():
    if False:
        for i in range(10):
            print('nop')
    return

def onExit():
    if False:
        return 10
    return

def onFrameStart(frame):
    if False:
        for i in range(10):
            print('nop')
    return

def onFrameEnd(frame):
    if False:
        return 10
    mqSender = parent.Guide.op('./mqout')
    pars = parent.mqguide.par
    tid = int(pars.Trackid.eval())
    gid = int(pars.Guideid.eval()) - 1
    x = float(pars.Positionx.eval())
    y = -float(pars.Positiony.eval())
    z = float(pars.Height.eval())
    msg = f'{x:.2f},{z:.2f},{y:.2f},{gid},Tracker:{tid}'
    mqSender.send(msg)
    return

def onPlayStateChange(state):
    if False:
        while True:
            i = 10
    return

def onDeviceChange():
    if False:
        return 10
    return

def onProjectPreSave():
    if False:
        i = 10
        return i + 15
    return

def onProjectPostSave():
    if False:
        for i in range(10):
            print('nop')
    return