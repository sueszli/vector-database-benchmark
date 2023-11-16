import sys
ROBOT_LISTENER_API_VERSION = 2

def start_keyword(name, attrs):
    if False:
        while True:
            i = 10
    sys.stdout.write('start keyword %s\n' % name)
    sys.stderr.write('start keyword %s\n' % name)

def end_keyword(name, attrs):
    if False:
        i = 10
        return i + 15
    sys.stdout.write('end keyword %s\n' % name)
    sys.stderr.write('end keyword %s\n' % name)