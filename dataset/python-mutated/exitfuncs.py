"""SCons.exitfuncs

Register functions which are executed when SCons exits for any reason.

"""
__revision__ = 'src/engine/SCons/exitfuncs.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import atexit
_exithandlers = []

def _run_exitfuncs():
    if False:
        print('Hello World!')
    'run any registered exit functions\n\n    _exithandlers is traversed in reverse order so functions are executed\n    last in, first out.\n    '
    while _exithandlers:
        (func, targs, kargs) = _exithandlers.pop()
        func(*targs, **kargs)

def register(func, *targs, **kargs):
    if False:
        for i in range(10):
            print('nop')
    'register a function to be executed upon normal program termination\n\n    func - function to be called at exit\n    targs - optional arguments to pass to func\n    kargs - optional keyword arguments to pass to func\n    '
    _exithandlers.append((func, targs, kargs))
atexit.register(_run_exitfuncs)