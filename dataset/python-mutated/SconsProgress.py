""" Progress bar for Scons compilation part.

This does only the interfacing with tracing and collection of information.

"""
from nuitka.Progress import closeProgressBar, enableProgressBar, reportProgressBar, setupProgressBar
from nuitka.Tracing import scons_logger

def enableSconsProgressBar():
    if False:
        for i in range(10):
            print('nop')
    enableProgressBar()
    import atexit
    atexit.register(closeSconsProgressBar)
_total = None
_current = 0
_stage = None

def setSconsProgressBarTotal(name, total):
    if False:
        i = 10
        return i + 15
    global _total, _stage
    _total = total
    _stage = name
    setupProgressBar(stage='%s C' % name, unit='file', total=total)

def updateSconsProgressBar():
    if False:
        for i in range(10):
            print('nop')
    global _current
    _current += 1
    reportProgressBar(item=None, update=True)
    if _current == _total:
        closeSconsProgressBar()
        scons_logger.info('%s linking program with %d files (no progress information available for this stage).' % (_stage, _total))

def closeSconsProgressBar():
    if False:
        i = 10
        return i + 15
    closeProgressBar()

def reportSlowCompilation(env, cmd, delta_time):
    if False:
        for i in range(10):
            print('nop')
    if _current != _total:
        scons_logger.info('Slow C compilation detected, used %.0fs so far, scalability problem.' % delta_time)
    elif env.orig_lto_mode == 'auto' and env.lto_mode:
        scons_logger.info('Slow C linking detected, used %.0fs so far, consider using \'--lto=no\' for faster linking, or \'--lto=yes"\' to disable this message. ' % delta_time)