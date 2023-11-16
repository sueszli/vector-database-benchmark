from builtins import _user_controlled

def rce_problem():
    if False:
        i = 10
        return i + 15
    x = _user_controlled()
    eval(x)