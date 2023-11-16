try:
    str.not_existing
except TypeError:
    pass
try:
    str.not_existing
except AttributeError:
    str.not_existing
    pass
try:
    import not_existing_import
except ImportError:
    pass
try:
    import not_existing_import2
except AttributeError:
    pass
try:
    str.not_existing
except (TypeError, AttributeError):
    pass
try:
    str.not_existing
except ImportError:
    pass
except (NotImplementedError, AttributeError):
    pass
try:
    str.not_existing
except (TypeError, NotImplementedError):
    pass
try:
    str.not_existing
except AttributeError:
    pass
try:
    str.not_existing
except [AttributeError]:
    pass
try:
    pass
except Undefined:
    pass
try:
    undefined
except Exception:
    pass
try:
    undefined
except:
    pass
if hasattr(str, 'undefined'):
    str.undefined
    str.upper
    str.undefined2
    int.undefined
else:
    str.upper
    str.undefined

def i_see(r):
    if False:
        return 10
    return r

def lala():
    if False:
        while True:
            i = 10
    a = TypeError
    try:
        i_see()
    except a:
        pass
    i_see()