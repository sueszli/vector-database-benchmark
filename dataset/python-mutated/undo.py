import itertools
from copy import copy
from visidata import vd, options, VisiData, BaseSheet, UNLOADED
BaseSheet.init('undone', list)
vd.option('undo', True, 'enable undo/redo')
nonUndo = 'commit open-file reload-sheet'.split()

def isUndoableCommand(longname):
    if False:
        while True:
            i = 10
    for n in nonUndo:
        if longname.startswith(n):
            return False
    return True

@VisiData.api
def addUndo(vd, undofunc, *args, **kwargs):
    if False:
        print('Hello World!')
    'On undo of latest command, call ``undofunc(*args, **kwargs)``.'
    if vd.options.undo:
        if getattr(vd, 'activeCommand', UNLOADED) is UNLOADED:
            return
        r = vd.modifyCommand
        if not r or not isUndoableCommand(r.longname) or (not vd.activeCommand) or (not vd.isLoggableCommand(vd.activeCommand.longname)):
            return
        if not r.undofuncs:
            r.undofuncs = []
        r.undofuncs.append((undofunc, args, kwargs))

@VisiData.api
def undo(vd, sheet):
    if False:
        while True:
            i = 10
    if not vd.options.undo:
        vd.fail('options.undo not enabled')
    for (i, cmdlogrow) in enumerate(sheet.cmdlog_sheet.rows[:0:-1]):
        if cmdlogrow.undofuncs:
            for (undofunc, args, kwargs) in cmdlogrow.undofuncs[::-1]:
                undofunc(*args, **kwargs)
            sheet.undone.append(cmdlogrow)
            row_idx = len(sheet.cmdlog_sheet.rows) - 1 - i
            del sheet.cmdlog_sheet.rows[row_idx]
            vd.clearCaches()
            vd.moveToReplayContext(cmdlogrow, sheet)
            vd.status('%s undone' % cmdlogrow.longname)
            return
    vd.fail('nothing to undo on current sheet')

@VisiData.api
def redo(vd, sheet):
    if False:
        for i in range(10):
            print('nop')
    sheet.undone or vd.fail('nothing to redo')
    cmdlogrow = sheet.undone.pop()
    vd.replayOne(cmdlogrow)
    vd.status('%s redone' % cmdlogrow.longname)

def undoAttrFunc(objs, attrname):
    if False:
        while True:
            i = 10
    'Return closure that sets attrname on each obj to its former value.'
    oldvals = [(o, getattr(o, attrname)) for o in objs]

    def _undofunc():
        if False:
            i = 10
            return i + 15
        for (o, v) in oldvals:
            setattr(o, attrname, v)
    return _undofunc

class Fanout(list):
    """Fan out attribute changes to every element in a list."""

    def __getattr__(self, k):
        if False:
            while True:
                i = 10
        return Fanout([getattr(o, k) for o in self])

    def __setattr__(self, k, v):
        if False:
            return 10
        vd.addUndo(undoAttrFunc(self, k))
        for o in self:
            setattr(o, k, v)

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return Fanout([o(*args, **kwargs) for o in self])

def undoAttrCopyFunc(objs, attrname):
    if False:
        i = 10
        return i + 15
    'Return closure that sets attrname on each obj to its former value.'
    oldvals = [(o, copy(getattr(o, attrname))) for o in objs]

    def _undofunc():
        if False:
            while True:
                i = 10
        for (o, v) in oldvals:
            setattr(o, attrname, v)
    return _undofunc

@VisiData.api
def addUndoSetValues(vd, cols, rows):
    if False:
        while True:
            i = 10
    'Add undo function to reset values for *rows* in *cols*.'
    oldvals = [(c, r, c.getValue(r)) for (c, r) in itertools.product(cols, vd.Progress(rows, gerund='doing'))]

    def _undo():
        if False:
            return 10
        for (c, r, v) in oldvals:
            c.setValue(r, v, setModified=False)
    vd.addUndo(_undo)

@VisiData.api
def addUndoColNames(vd, cols):
    if False:
        i = 10
        return i + 15
    oldnames = [(c, c.name) for c in cols]

    def _undo():
        if False:
            for i in range(10):
                print('nop')
        for (c, name) in oldnames:
            c.name = name
    vd.addUndo(_undo)
BaseSheet.addCommand('U', 'undo-last', 'vd.undo(sheet)', 'Undo the most recent change (options.undo must be enabled)')
BaseSheet.addCommand('R', 'redo-last', 'vd.redo(sheet)', 'Redo the most recent undo (options.undo must be enabled)')
vd.addGlobals(undoAttrFunc=undoAttrFunc, Fanout=Fanout, undoAttrCopyFunc=undoAttrCopyFunc)
vd.addMenuItems('\n    Edit > Undo > undo-last\n    Edit > Redo > redo-last\n')