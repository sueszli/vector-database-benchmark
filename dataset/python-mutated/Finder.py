"""Contains various utility functions."""
__all__ = ['findClass', 'rebindClass', 'copyFuncs', 'replaceMessengerFunc', 'replaceTaskMgrFunc', 'replaceStateFunc', 'replaceCRFunc', 'replaceAIRFunc', 'replaceIvalFunc']
import types
import os
import sys
from direct.showbase.MessengerGlobal import messenger
from direct.task.TaskManagerGlobal import taskMgr

def findClass(className):
    if False:
        i = 10
        return i + 15
    '\n    Look in sys.modules dictionary for a module that defines a class\n    with this className.\n    '
    for (moduleName, module) in sys.modules.items():
        if module:
            classObj = module.__dict__.get(className)
            if classObj and isinstance(classObj, type) and (classObj.__module__ == moduleName):
                return [classObj, module.__dict__]
    return None

def rebindClass(filename):
    if False:
        for i in range(10):
            print('nop')
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        if line[0:6] == 'class ':
            classHeader = line[6:].strip()
            parenLoc = classHeader.find('(')
            if parenLoc > 0:
                className = classHeader[:parenLoc]
            else:
                colonLoc = classHeader.find(':')
                if colonLoc > 0:
                    className = classHeader[:colonLoc]
                else:
                    print('error: className not found')
                    file.close()
                    os.remove(filename)
                    return
            print('Rebinding class name: ' + className)
            break
    res = findClass(className)
    if not res:
        print('Warning: Finder could not find class')
        file.close()
        os.remove(filename)
        return
    (realClass, realNameSpace) = res
    exec(compile(open(filename).read(), filename, 'exec'), realNameSpace)
    tmpClass = realNameSpace[className]
    copyFuncs(tmpClass, realClass)
    realNameSpace[className] = realClass
    file.close()
    os.remove(filename)
    print('    Finished rebind')

def copyFuncs(fromClass, toClass):
    if False:
        i = 10
        return i + 15
    replaceFuncList = []
    newFuncList = []
    for (funcName, newFunc) in fromClass.__dict__.items():
        if isinstance(newFunc, types.FunctionType):
            oldFunc = toClass.__dict__.get(funcName)
            if oldFunc:
                replaceFuncList.append((oldFunc, funcName, newFunc))
            else:
                newFuncList.append((funcName, newFunc))
    replaceMessengerFunc(replaceFuncList)
    replaceTaskMgrFunc(replaceFuncList)
    replaceStateFunc(replaceFuncList)
    replaceCRFunc(replaceFuncList)
    replaceAIRFunc(replaceFuncList)
    replaceIvalFunc(replaceFuncList)
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        setattr(toClass, funcName, newFunc)
    for (funcName, newFunc) in newFuncList:
        setattr(toClass, funcName, newFunc)

def replaceMessengerFunc(replaceFuncList):
    if False:
        print('Hello World!')
    try:
        messenger
    except Exception:
        return
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        res = messenger.replaceMethod(oldFunc, newFunc)
        if res:
            print('replaced %s messenger function(s): %s' % (res, funcName))

def replaceTaskMgrFunc(replaceFuncList):
    if False:
        for i in range(10):
            print('nop')
    try:
        taskMgr
    except Exception:
        return
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        if taskMgr.replaceMethod(oldFunc, newFunc):
            print('replaced taskMgr function: %s' % funcName)

def replaceStateFunc(replaceFuncList):
    if False:
        return 10
    if not sys.modules.get('base.direct.fsm.State'):
        return
    from direct.fsm.State import State
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        res = State.replaceMethod(oldFunc, newFunc)
        if res:
            print('replaced %s FSM transition function(s): %s' % (res, funcName))

def replaceCRFunc(replaceFuncList):
    if False:
        for i in range(10):
            print('nop')
    try:
        base.cr
    except Exception:
        return
    if hasattr(base.cr, 'isFake'):
        return
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        if base.cr.replaceMethod(oldFunc, newFunc):
            print('replaced DistributedObject function: %s' % funcName)

def replaceAIRFunc(replaceFuncList):
    if False:
        return 10
    try:
        simbase.air
    except Exception:
        return
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        if simbase.air.replaceMethod(oldFunc, newFunc):
            print('replaced DistributedObject function: %s' % funcName)

def replaceIvalFunc(replaceFuncList):
    if False:
        print('Hello World!')
    if not sys.modules.get('base.direct.interval.IntervalManager'):
        return
    from direct.interval.FunctionInterval import FunctionInterval
    for (oldFunc, funcName, newFunc) in replaceFuncList:
        res = FunctionInterval.replaceMethod(oldFunc, newFunc)
        if res:
            print('replaced %s interval function(s): %s' % (res, funcName))