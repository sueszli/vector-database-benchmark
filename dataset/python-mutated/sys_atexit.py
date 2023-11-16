import sys
try:
    sys.atexit
except AttributeError:
    print('SKIP')
    raise SystemExit
some_var = None

def do_at_exit():
    if False:
        i = 10
        return i + 15
    print('done at exit:', some_var)
sys.atexit(do_at_exit)
some_var = 'ok'
print('done before exit')