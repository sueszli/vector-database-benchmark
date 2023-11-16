import inspect
import os
import sys
import scapy

def linkcode_resolve(domain, info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine the URL corresponding to Python object\n    '
    if domain != 'py':
        return None
    modname = info['module']
    fullname = info['fullname']
    submod = sys.modules.get(modname)
    if submod is None:
        return None
    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)
    fn = None
    lineno = None
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None
    try:
        (source, lineno) = inspect.getsourcelines(obj)
    except Exception:
        lineno = None
    fn = os.path.relpath(fn, start=os.path.dirname(scapy.__file__))
    if lineno:
        linespec = '#L%d-L%d' % (lineno, lineno + len(source) - 1)
    else:
        linespec = ''
    if 'dev' in scapy.__version__:
        return 'https://github.com/secdev/scapy/blob/master/scapy/%s%s' % (fn, linespec)
    else:
        return 'https://github.com/secdev/scapy/blob/v%s/scapy/%s%s' % (scapy.__version__, fn, linespec)