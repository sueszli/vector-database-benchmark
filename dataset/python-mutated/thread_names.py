import time
import threading

def main():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        th = threading.Thread(target=lambda : time.sleep(10000))
        th.name = 'CustomThreadName-%s' % i
        th.start()
    time.sleep(10000)
if __name__ == '__main__':
    main()