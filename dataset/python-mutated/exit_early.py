import argparse
import os
import sys
import time

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--silent', action='store_true')
    args = parser.parse_args()
    if not args.silent:
        sys.stdout.write('%d\n' % (os.getpid(),))
        sys.stdout.flush()
    max_time = time.time() + 2
    while True:
        time.sleep(0.1)
        target = time.time() + 0.1
        while True:
            now = time.time()
            if now >= max_time:
                return
            if now >= target:
                break
if __name__ == '__main__':
    main()