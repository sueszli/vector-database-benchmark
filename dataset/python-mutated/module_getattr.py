this = __import__(__name__)
try:
    this.does_not_exist
    assert False
except AttributeError:
    pass

def __getattr__(attr):
    if False:
        return 10
    if attr == 'does_not_exist':
        return False
    raise AttributeError
if not hasattr(this, 'does_not_exist'):
    print('SKIP')
    raise SystemExit
print(this.does_not_exist)