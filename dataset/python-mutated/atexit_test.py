try:
    import atexit
except ImportError:
    print('SKIP')
    raise SystemExit

@atexit.register
def skip_at_exit():
    if False:
        while True:
            i = 10
    print('skip at exit')

@atexit.register
def do_at_exit(*args, **kwargs):
    if False:
        return 10
    print('done at exit:', args, kwargs)
atexit.unregister(skip_at_exit)
atexit.register(do_at_exit, 'ok', 1, arg='2', param=(3, 4))
print('done before exit')