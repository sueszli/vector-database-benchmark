import sys
from . import utils

def install_shims():
    if False:
        while True:
            i = 10
    'Install shims to fix version incompatibility'
    install_bottle_import_redirect_shim()

def install_bottle_import_redirect_shim():
    if False:
        print('Hello World!')
    "\n    https://github.com/brentvollebregt/auto-py-to-exe/issues/433 explains that a ModuleNotFoundError is raised when trying\n    to import bottle extensions using Python 3.12.\n    This shim will patch this issue with some code that is currently on bottle's main branch.\n    This shim is only needed on Python versions >=3.12 and bottle versions <=0.12.25 (hoping the next version fixes this issue)\n    "
    if sys.version_info < (3, 12):
        return
    import bottle
    if utils.parse_version_tuple(bottle.__version__) > (0, 12, 25):
        return
    if hasattr(bottle._ImportRedirect, 'find_spec'):
        return
    print(f'Warning: Installing shim for bottle import redirects (using Python={sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]} and bottle={bottle.__version__})')

    def find_spec(self, fullname, path, target=None):
        if False:
            return 10
        if '.' not in fullname:
            return
        if fullname.rsplit('.', 1)[0] != self.name:
            return
        from importlib.util import spec_from_loader
        return spec_from_loader(fullname, self)
    bottle._ImportRedirect.find_spec = find_spec