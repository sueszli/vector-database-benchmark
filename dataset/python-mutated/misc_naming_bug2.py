from x import some_fetch
GLOBAL_CONST = 'SOME_CONST'

def fetch_global_const():
    if False:
        i = 10
        return i + 15
    return some_fetch(GLOBAL_CONST)