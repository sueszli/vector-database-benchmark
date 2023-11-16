from __future__ import absolute_import
import os
import signal
_breakin_signal_number = None
_breakin_signal_name = None

def _debug(signal_number, interrupted_frame):
    if False:
        for i in range(10):
            print('nop')
    import pdb
    import sys
    sys.stderr.write("** %s received, entering debugger\n** Type 'c' to continue or 'q' to stop the process\n** Or %s again to quit (and possibly dump core)\n" % (_breakin_signal_name, _breakin_signal_name))
    sys.stderr.flush()
    signal.signal(_breakin_signal_number, signal.SIG_DFL)
    try:
        pdb.set_trace()
    finally:
        signal.signal(_breakin_signal_number, _debug)

def determine_signal():
    if False:
        return 10
    global _breakin_signal_number
    global _breakin_signal_name
    if _breakin_signal_number is not None:
        return _breakin_signal_number
    sigquit = getattr(signal, 'SIGQUIT', None)
    sigbreak = getattr(signal, 'SIGBREAK', None)
    if sigquit is not None:
        _breakin_signal_number = sigquit
        _breakin_signal_name = 'SIGQUIT'
    elif sigbreak is not None:
        _breakin_signal_number = sigbreak
        _breakin_signal_name = 'SIGBREAK'
    return _breakin_signal_number

def hook_debugger_to_signal():
    if False:
        for i in range(10):
            print('nop')
    'Add a signal handler so we drop into the debugger.\n\n    On Unix, this is hooked into SIGQUIT (C-\\), and on Windows, this is\n    hooked into SIGBREAK (C-Pause).\n    '
    if os.environ.get('BZR_SIGQUIT_PDB', '1') == '0':
        return
    sig = determine_signal()
    if sig is None:
        return
    signal.signal(sig, _debug)