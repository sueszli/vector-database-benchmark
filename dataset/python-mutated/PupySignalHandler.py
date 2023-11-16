import signal
winch_handler = None

def set_signal_winch(handler):
    if False:
        i = 10
        return i + 15
    ' return the old signal handler '
    global winch_handler
    old_handler = winch_handler
    winch_handler = handler
    return old_handler

def signal_winch(signum, frame):
    if False:
        while True:
            i = 10
    global winch_handler
    if winch_handler:
        return winch_handler(signum, frame)
signal.signal(signal.SIGWINCH, signal_winch)