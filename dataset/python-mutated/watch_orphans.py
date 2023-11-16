import os
import signal
import sys
import time

def watch(args):
    if False:
        i = 10
        return i + 15
    if not args or len(args) != 2:
        return
    parent_pid = int(args[0])
    tail_pid = int(args[1])
    if not parent_pid or not tail_pid:
        return
    while True:
        if os.getppid() != parent_pid:
            try:
                os.kill(tail_pid, signal.SIGTERM)
            except OSError:
                pass
            break
        else:
            time.sleep(1)
if __name__ == '__main__':
    watch(sys.argv[1:])