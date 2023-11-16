"""Test PyMongo with a variety of greenlet-based monkey-patching frameworks."""
from __future__ import annotations
import getopt
import sys
import pytest

def run_gevent():
    if False:
        while True:
            i = 10
    'Prepare to run tests with Gevent. Can raise ImportError.'
    from gevent import monkey
    monkey.patch_all()

def run_eventlet():
    if False:
        print('Hello World!')
    'Prepare to run tests with Eventlet. Can raise ImportError.'
    import eventlet
    eventlet.sleep()
    eventlet.monkey_patch()
FRAMEWORKS = {'gevent': run_gevent, 'eventlet': run_eventlet}

def list_frameworks():
    if False:
        i = 10
        return i + 15
    'Tell the user what framework names are valid.'
    sys.stdout.write('Testable frameworks: %s\n\nNote that membership in this list means the framework can be tested with\nPyMongo, not necessarily that it is officially supported.\n' % ', '.join(sorted(FRAMEWORKS)))

def run(framework_name, *args):
    if False:
        print('Hello World!')
    'Run tests with monkey-patching enabled. Can raise ImportError.'
    FRAMEWORKS[framework_name]()
    sys.exit(pytest.main(list(args)))

def main():
    if False:
        return 10
    'Parse options and run tests.'
    usage = f'python {sys.argv[0]} FRAMEWORK_NAME\n\nTest PyMongo with a variety of greenlet-based monkey-patching frameworks. See\npython {sys.argv[0]} --help-frameworks.'
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'h', ['help', 'help-frameworks'])
    except getopt.GetoptError as err:
        print(str(err))
        print(usage)
        sys.exit(2)
    for (option_name, _) in opts:
        if option_name in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif option_name == '--help-frameworks':
            list_frameworks()
            sys.exit()
        else:
            raise AssertionError('unhandled option')
    if not args:
        print(usage)
        sys.exit(1)
    if args[0] not in FRAMEWORKS:
        print('%r is not a testable framework.\n' % args[0])
        list_frameworks()
        sys.exit(1)
    run(args[0], *args[1:])
if __name__ == '__main__':
    main()