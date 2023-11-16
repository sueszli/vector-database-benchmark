ROBOT_LISTENER_API_VERSION = 2
from robot.libraries.BuiltIn import BuiltIn
run_keyword = BuiltIn().run_keyword

def start_suite(name, attrs):
    if False:
        return 10
    run_keyword('Log', 'start_suite')

def end_suite(name, attrs):
    if False:
        return 10
    run_keyword('Log', 'end_suite')

def start_test(name, attrs):
    if False:
        i = 10
        return i + 15
    run_keyword('Log', 'start_test')

def end_test(name, attrs):
    if False:
        for i in range(10):
            print('nop')
    run_keyword('Log', 'end_test')

def start_keyword(name, attrs):
    if False:
        print('Hello World!')
    run_keyword('Log', 'start_keyword')

def end_keyword(name, attrs):
    if False:
        return 10
    run_keyword('Log', 'end_keyword')