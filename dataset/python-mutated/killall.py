"""Kill a process by name."""
import os
import sys
import psutil

def main():
    if False:
        i = 10
        return i + 15
    if len(sys.argv) != 2:
        sys.exit('usage: %s name' % __file__)
    else:
        NAME = sys.argv[1]
    killed = []
    for proc in psutil.process_iter():
        if proc.name() == NAME and proc.pid != os.getpid():
            proc.kill()
            killed.append(proc.pid)
    if not killed:
        sys.exit('%s: no process found' % NAME)
    else:
        sys.exit(0)
if __name__ == '__main__':
    main()