from robot.api import Failure, Error

def failure(msg='I failed my duties', html=False):
    if False:
        i = 10
        return i + 15
    raise Failure(msg, html)

def error(msg='I errored my duties', html=False):
    if False:
        i = 10
        return i + 15
    raise Error(msg, html=html)