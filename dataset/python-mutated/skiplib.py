from robot.api import SkipExecution

class CustomSkipException(Exception):
    ROBOT_SKIP_EXECUTION = True

def skip_with_message(msg, html=False):
    if False:
        while True:
            i = 10
    raise SkipExecution(msg, html)

def skip_with_custom_exception():
    if False:
        for i in range(10):
            print('nop')
    raise CustomSkipException('Skipped with custom exception.')