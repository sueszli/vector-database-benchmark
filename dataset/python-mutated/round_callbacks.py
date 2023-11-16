def onInitialize(timerOp):
    if False:
        for i in range(10):
            print('nop')
    return 0

def onReady(timerOp):
    if False:
        for i in range(10):
            print('nop')
    return

def onStart(timerOp):
    if False:
        return 10
    op.Roundtimer.par.Roundlock = 0
    op.Tracker.StartRound()
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
        i = 10
        return i + 15
    return

def onCycleStart(timerOp, segment, cycle):
    if False:
        print('Hello World!')
    return

def onCycleEndAlert(timerOp, segment, cycle, alertSegment, alertDone, interrupt):
    if False:
        return 10
    op.Roundtimer.par.Roundlock.val = 1
    op.Sound.SendCountdown()
    return

def onCycle(timerOp, segment, cycle):
    if False:
        for i in range(10):
            print('nop')
    return

def onDone(timerOp, segment, interrupt):
    if False:
        for i in range(10):
            print('nop')
    op.Tracker.StopRound()
    op.Sound.SendEvaluationStart()
    return