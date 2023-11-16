import _common
from trio._core._multierror import MultiError

def exc1_fn() -> Exception:
    if False:
        return 10
    try:
        raise ValueError
    except Exception as exc:
        return exc

def exc2_fn() -> Exception:
    if False:
        for i in range(10):
            print('nop')
    try:
        raise KeyError
    except Exception as exc:
        return exc
raise MultiError([exc1_fn(), exc2_fn()])