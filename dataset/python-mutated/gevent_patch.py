import os
from typing import Optional
from gevent import monkey

def str_to_bool(bool_str: Optional[str]) -> bool:
    if False:
        print('Hello World!')
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == 'true' or bool_str == '1':
        result = True
    return result
GEVENT_MONKEYPATCH = str_to_bool(os.environ.get('GEVENT_MONKEYPATCH', 'False'))

def is_notebook() -> bool:
    if False:
        print('Hello World!')
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False
jupyter_notebook = is_notebook()
if jupyter_notebook:
    monkey.patch_all(thread=False)