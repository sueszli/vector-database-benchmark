from time import sleep as _sleep
import sys
absolute_import = sys.version_info[0] >= 3
if absolute_import:
    exec('from . import db')
else:
    import db
_deadlock_MinSleepTime = 1.0 / 128
_deadlock_MaxSleepTime = 3.14159
_deadlock_VerboseFile = None

def DeadlockWrap(function, *_args, **_kwargs):
    if False:
        while True:
            i = 10
    'DeadlockWrap(function, *_args, **_kwargs) - automatically retries\n    function in case of a database deadlock.\n\n    This is a function intended to be used to wrap database calls such\n    that they perform retrys with exponentially backing off sleeps in\n    between when a DBLockDeadlockError exception is raised.\n\n    A \'max_retries\' parameter may optionally be passed to prevent it\n    from retrying forever (in which case the exception will be reraised).\n\n        d = DB(...)\n        d.open(...)\n        DeadlockWrap(d.put, "foo", data="bar")  # set key "foo" to "bar"\n    '
    sleeptime = _deadlock_MinSleepTime
    max_retries = _kwargs.get('max_retries', -1)
    if 'max_retries' in _kwargs:
        del _kwargs['max_retries']
    while True:
        try:
            return function(*_args, **_kwargs)
        except db.DBLockDeadlockError:
            if _deadlock_VerboseFile:
                _deadlock_VerboseFile.write('dbutils.DeadlockWrap: sleeping %1.3f\n' % sleeptime)
            _sleep(sleeptime)
            sleeptime *= 2
            if sleeptime > _deadlock_MaxSleepTime:
                sleeptime = _deadlock_MaxSleepTime
            max_retries -= 1
            if max_retries == -1:
                raise