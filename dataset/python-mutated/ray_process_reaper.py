import atexit
import os
import signal
import sys
import time
'\nThis is a lightweight "reaper" process used to ensure that ray processes are\ncleaned up properly when the main ray process dies unexpectedly (e.g.,\nsegfaults or gets SIGKILLed). Note that processes may not be cleaned up\nproperly if this process is SIGTERMed or SIGKILLed.\n\nIt detects that its parent has died by reading from stdin, which must be\ninherited from the parent process so that the OS will deliver an EOF if the\nparent dies. When this happens, the reaper process kills the rest of its\nprocess group (first attempting graceful shutdown with SIGTERM, then escalating\nto SIGKILL).\n'
SIGTERM_GRACE_PERIOD_SECONDS = 1

def reap_process_group(*args):
    if False:
        print('Hello World!')

    def sigterm_handler(*args):
        if False:
            i = 10
            return i + 15
        time.sleep(SIGTERM_GRACE_PERIOD_SECONDS)
        if sys.platform == 'win32':
            atexit.unregister(sigterm_handler)
            os.kill(0, signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(0, signal.SIGKILL)
    if sys.platform == 'win32':
        atexit.register(sigterm_handler)
    else:
        signal.signal(signal.SIGTERM, sigterm_handler)
    if sys.platform == 'win32':
        os.kill(0, signal.CTRL_C_EVENT)
    else:
        os.killpg(0, signal.SIGTERM)

def main():
    if False:
        print('Hello World!')
    while len(sys.stdin.read()) != 0:
        pass
    reap_process_group()
if __name__ == '__main__':
    main()