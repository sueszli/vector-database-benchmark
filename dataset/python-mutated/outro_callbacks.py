def onInitialize(timerOp):
    if False:
        return 10
    return 0

def onReady(timerOp):
    if False:
        i = 10
        return i + 15
    return

def onStart(timerOp):
    if False:
        print('Hello World!')
    return

def onTimerPulse(timerOp, segment):
    if False:
        return 10
    return

def whileTimerActive(timerOp, segment, cycle, fraction):
    if False:
        i = 10
        return i + 15
    return

def onSegmentEnter(timerOp, segment, interrupt):
    if False:
        print('Hello World!')
    return

def onSegmentExit(timerOp, segment, interrupt):
    if False:
        i = 10
        return i + 15
    return

def onCycleStart(timerOp, segment, cycle):
    if False:
        i = 10
        return i + 15
    if cycle < 3:
        mode = 'high'
    else:
        mode = 'low'
    op.Sound.SendEvaluationRank(mode)
    return

def onCycleEndAlert(timerOp, segment, cycle, alertSegment, alertDone, interrupt):
    if False:
        return 10
    return

def onCycle(timerOp, segment, cycle):
    if False:
        print('Hello World!')
    return

def onDone(timerOp, segment, interrupt):
    if False:
        while True:
            i = 10
    return