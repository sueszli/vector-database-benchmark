"""
Proxy for import module, which can be used to catch dynamic imports which wasn't added to HIDDEN_IMPORTS
to pyinstaller configuration
"""
import importlib
import logging
from samcli.cli import hidden_imports
LOG = logging.getLogger(__name__)
_original_import = importlib.import_module

class MissingDynamicImportError(ImportError):
    """
    Thrown when a dynamic import is used without adding it into hidden imports constant
    """

def _dynamic_import(name, package=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replaces original import_module function and then analyzes all the imports going through this call.\n    If the package is not defined in hidden imports, then it will raise an error\n    '
    for hidden_import in hidden_imports.SAM_CLI_HIDDEN_IMPORTS:
        if name == hidden_import or name.startswith(f'{hidden_import}.'):
            LOG.debug('Importing a package which was already defined in hidden imports name: %s, package: %s', name, package)
            return _original_import(name, package)
    LOG.error('Dynamic import (name: %s package: %s) which is not defined in hidden imports: %s', name, package, hidden_imports.SAM_CLI_HIDDEN_IMPORTS)
    raise MissingDynamicImportError(f'Dynamic import not allowed for name: {name} package: {package}')

def attach_import_module_proxy():
    if False:
        print('Hello World!')
    '\n    Attaches import_module proxy which will analyze every dynamic import and raise an error if it is not defined in\n    hidden imports configuration\n    '
    importlib.import_module = _dynamic_import