from robot.errors import PassExecution
from robot.libraries.BuiltIn import BuiltIn

def raise_pass_execution_exception(msg):
    if False:
        return 10
    raise PassExecution(msg)

def call_pass_execution_method(msg):
    if False:
        i = 10
        return i + 15
    BuiltIn().pass_execution(msg, 'lol')