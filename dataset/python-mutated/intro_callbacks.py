def onInitialize(timerOp):
    if False:
        i = 10
        return i + 15
    return 0

def onReady(timerOp):
    if False:
        print('Hello World!')
    return

def onStart(timerOp):
    if False:
        for i in range(10):
            print('nop')
    name = str(op.Scene.par.Ident.eval())
    op.Sound.SendScene(name)
    op.Sound.SendIntro(name)
    op.Tracker.Initround()
    return

def onTimerPulse(timerOp, segment):
    if False:
        i = 10
        return i + 15
    return

def whileTimerActive(timerOp, segment, cycle, fraction):
    if False:
        for i in range(10):
            print('nop')
    return

def onSegmentEnter(timerOp, segment, interrupt):
    if False:
        i = 10
        return i + 15
    return

def onSegmentExit(timerOp, segment, interrupt):
    if False:
        while True:
            i = 10
    return

def onCycleStart(timerOp, segment, cycle):
    if False:
        while True:
            i = 10
    return

def onCycleEndAlert(timerOp, segment, cycle, alertSegment, alertDone, interrupt):
    if False:
        return 10
    return

def onCycle(timerOp, segment, cycle):
    if False:
        return 10
    return

def onDone(timerOp, segment, interrupt):
    if False:
        return 10
    return