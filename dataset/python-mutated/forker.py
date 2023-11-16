import argparse
import os
import time

def spawn(count):
    if False:
        while True:
            i = 10
    t0 = time.time()
    x = 0
    while time.time() < t0 + 0.1:
        x += 1
    if count:
        pid = os.fork()
        if pid == 0:
            spawn(count - 1)
        else:
            os.waitpid(pid, 0)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=5, help='How many times to fork')
    args = parser.parse_args()
    pid = os.fork()
    if pid == 0:
        spawn(args.count)
    else:
        os.waitpid(pid, 0)
if __name__ == '__main__':
    main()