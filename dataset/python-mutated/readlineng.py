"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
_readline = None
try:
    from readline import *
    import readline as _readline
except:
    try:
        from pyreadline import *
        import pyreadline as _readline
    except:
        pass
from lib.core.data import logger
from lib.core.settings import IS_WIN
from lib.core.settings import PLATFORM
if IS_WIN and _readline:
    try:
        _outputfile = _readline.GetOutputFile()
    except AttributeError:
        debugMsg = "Failed GetOutputFile when using platform's "
        debugMsg += 'readline library'
        logger.debug(debugMsg)
        _readline = None
uses_libedit = False
if PLATFORM == 'mac' and _readline:
    import commands
    (status, result) = commands.getstatusoutput('otool -L %s | grep libedit' % _readline.__file__)
    if status == 0 and len(result) > 0:
        _readline.parse_and_bind('bind ^I rl_complete')
        debugMsg = "Leopard libedit detected when using platform's "
        debugMsg += 'readline library'
        logger.debug(debugMsg)
        uses_libedit = True
if _readline:
    if not hasattr(_readline, 'clear_history'):

        def clear_history():
            if False:
                return 10
            pass
        _readline.clear_history = clear_history