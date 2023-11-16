from robot.libraries.BuiltIn import BuiltIn
BIN = BuiltIn()
ROBOT_LISTENER_API_VERSION = 2

def start_keyword(*args):
    if False:
        while True:
            i = 10
    if BIN.get_variables()['${TESTNAME}'] == 'Listener Using BuiltIn':
        BIN.set_test_variable('${SET BY LISTENER}', 'quux')