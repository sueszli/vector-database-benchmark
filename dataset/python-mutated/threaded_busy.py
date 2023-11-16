import os
import sys
import threading

def do_sleep():
    if False:
        print('Hello World!')
    while True:
        pass

def main():
    if False:
        print('Hello World!')
    sys.stdout.write('%d\n' % (os.getpid(),))
    sys.stdout.flush()
    thread = threading.Thread(target=do_sleep)
    thread.start()
    do_sleep()
if __name__ == '__main__':
    main()