try:
    Exception().__cause__
except AttributeError:
    print('SKIP')
    raise SystemExit
try:
    import traceback
except:
    print('SKIP')
    raise SystemExit

def print_exc_info(e, chain=True):
    if False:
        return 10
    print('-' * 72)
    traceback.print_exception(None, e, e.__traceback__, chain=chain)
    print('-' * 72)
    print()
try:
    try:
        1 / 0
    except Exception as inner:
        raise RuntimeError() from inner
except Exception as e:
    print_exc_info(e, chain=False)
    print_exc_info(e)
print()
try:
    try:
        1 / 0
    except Exception as inner:
        raise RuntimeError() from OSError()
except Exception as e:
    print_exc_info(e)
print()
try:
    try:
        1 / 0
    except Exception as inner:
        raise RuntimeError()
except Exception as e:
    print_exc_info(e)
print()
try:
    try:
        1 / 0
    except Exception as inner:
        raise RuntimeError() from None
except Exception as e:
    print_exc_info(e)
try:
    try:
        raise RuntimeError()
    except Exception as inner:
        1 / 0
except Exception as e:
    print_exc_info(e)
try:
    try:
        1 / 0
    except Exception as inner:
        raise inner
except Exception as e:
    print_exc_info(e, chain=False)
    print_exc_info(e)
print()