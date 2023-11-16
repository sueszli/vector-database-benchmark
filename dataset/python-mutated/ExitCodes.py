EX_OK = 0
EX_GENERAL = 1
EX_PARTIAL = 2
EX_SERVERMOVED = 10
EX_SERVERERROR = 11
EX_NOTFOUND = 12
EX_CONFLICT = 13
EX_PRECONDITION = 14
EX_SERVICE = 15
EX_USAGE = 64
EX_DATAERR = 65
EX_SOFTWARE = 70
EX_OSERR = 71
EX_OSFILE = 72
EX_IOERR = 74
EX_TEMPFAIL = 75
EX_ACCESSDENIED = 77
EX_CONFIG = 78
EX_CONNECTIONREFUSED = 111
_EX_SIGNAL = 128
_EX_SIGINT = 2
EX_BREAK = _EX_SIGNAL + _EX_SIGINT

class ExitScoreboard(object):
    """Helper to return best return code"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._success = 0
        self._notfound = 0
        self._failed = 0

    def success(self):
        if False:
            for i in range(10):
                print('nop')
        self._success += 1

    def notfound(self):
        if False:
            i = 10
            return i + 15
        self._notfound += 1

    def failed(self):
        if False:
            for i in range(10):
                print('nop')
        self._failed += 1

    def rc(self):
        if False:
            i = 10
            return i + 15
        if self._success:
            if not self._failed and (not self._notfound):
                return EX_OK
            elif self._failed:
                return EX_PARTIAL
        elif self._failed:
            return EX_GENERAL
        elif self._notfound:
            return EX_NOTFOUND
        return EX_GENERAL