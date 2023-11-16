from panda3d.core import VBase4
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr

def ROUND_TO(value, divisor):
    if False:
        for i in range(10):
            print('nop')
    return round(value / float(divisor)) * divisor

def ROUND_INT(val):
    if False:
        for i in range(10):
            print('nop')
    return int(round(val))

def CLAMP(val, minVal, maxVal):
    if False:
        i = 10
        return i + 15
    return min(max(val, minVal), maxVal)

def getTkColorString(color):
    if False:
        while True:
            i = 10
    '\n    Print out a Tk compatible version of a color string\n    '

    def toHex(intVal):
        if False:
            while True:
                i = 10
        val = int(intVal)
        if val < 16:
            return '0' + hex(val)[2:]
        else:
            return hex(val)[2:]
    r = toHex(color[0])
    g = toHex(color[1])
    b = toHex(color[2])
    return '#' + r + g + b

def lerpBackgroundColor(r, g, b, duration):
    if False:
        return 10
    '\n    Function to lerp background color to a new value\n    '

    def lerpColor(state):
        if False:
            return 10
        dt = base.clock.getDt()
        state.time += dt
        sf = state.time / state.duration
        if sf >= 1.0:
            base.setBackgroundColor(state.ec[0], state.ec[1], state.ec[2])
            return Task.done
        else:
            r = sf * state.ec[0] + (1 - sf) * state.sc[0]
            g = sf * state.ec[1] + (1 - sf) * state.sc[1]
            b = sf * state.ec[2] + (1 - sf) * state.sc[2]
            base.setBackgroundColor(r, g, b)
            return Task.cont
    taskMgr.remove('lerpBackgroundColor')
    t = taskMgr.add(lerpColor, 'lerpBackgroundColor')
    t.time = 0.0
    t.duration = duration
    t.sc = base.getBackgroundColor()
    t.ec = VBase4(r, g, b, 1)

def useDirectRenderStyle(nodePath, priority=0):
    if False:
        return 10
    '\n    Function to force a node path to use direct render style:\n    no lighting, and no wireframe\n    '
    nodePath.setLightOff(priority)
    nodePath.setRenderModeFilled()

def getFileData(filename, separator=','):
    if False:
        for i in range(10):
            print('nop')
    '\n    Open the specified file and strip out unwanted whitespace and\n    empty lines.  Return file as list of lists, one file line per element,\n    list elements based upon separator\n    '
    f = open(filename.toOsSpecific(), 'r')
    rawData = f.readlines()
    f.close()
    fileData = []
    for line in rawData:
        l = line.strip()
        if l:
            data = [s.strip() for s in l.split(separator)]
            fileData.append(data)
    return fileData