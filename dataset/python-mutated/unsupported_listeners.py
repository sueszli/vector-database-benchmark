import sys

def close():
    if False:
        for i in range(10):
            print('nop')
    sys.exit('This should not be called')

class V1ClassListener:
    ROBOT_LISTENER_API_VERSION = 1

    def close(self):
        if False:
            while True:
                i = 10
        close()

class InvalidVersionClassListener:
    ROBOT_LISTENER_API_VERSION = 'kekkonen'

    def close(self):
        if False:
            i = 10
            return i + 15
        close()