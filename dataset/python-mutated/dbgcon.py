DBGSTATE_NOT_DEBUGGING = 0
DBGSTATE_RUNNING = 1
DBGSTATE_BREAK = 2
DBGSTATE_QUITTING = 3
LINESTATE_CURRENT = 1
LINESTATE_BREAKPOINT = 2
LINESTATE_CALLSTACK = 4
OPT_HIDE = 'hide'
OPT_STOP_EXCEPTIONS = 'stopatexceptions'
import win32api
import win32ui

def DoGetOption(optsDict, optName, default):
    if False:
        print('Hello World!')
    optsDict[optName] = win32ui.GetProfileVal('Debugger Options', optName, default)

def LoadDebuggerOptions():
    if False:
        i = 10
        return i + 15
    opts = {}
    DoGetOption(opts, OPT_HIDE, 0)
    DoGetOption(opts, OPT_STOP_EXCEPTIONS, 1)
    return opts

def SaveDebuggerOptions(opts):
    if False:
        return 10
    for (key, val) in opts.items():
        win32ui.WriteProfileVal('Debugger Options', key, val)