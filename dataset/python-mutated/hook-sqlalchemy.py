import re
from PyInstaller import isolated
from PyInstaller.lib.modulegraph.modulegraph import SourceModule
from PyInstaller.lib.modulegraph.util import guess_encoding
from PyInstaller.utils.hooks import check_requirement, logger
excludedimports = ['sqlalchemy.testing']
hiddenimports = ['pysqlite2', 'MySQLdb', 'psycopg2', 'sqlalchemy.ext.baked']
if check_requirement('sqlalchemy >= 1.4'):
    hiddenimports.append('sqlalchemy.sql.default_comparator')

@isolated.decorate
def _get_dialect_modules(module_name):
    if False:
        while True:
            i = 10
    import importlib
    module = importlib.import_module(module_name)
    return [f'{module_name}.{submodule_name}' for submodule_name in module.__all__]
if check_requirement('sqlalchemy >= 0.6'):
    hiddenimports += _get_dialect_modules('sqlalchemy.dialects')
else:
    hiddenimports += _get_dialect_modules('sqlalchemy.databases')

def hook(hook_api):
    if False:
        for i in range(10):
            print('nop')
    '\n    SQLAlchemy 0.9 introduced the decorator \'util.dependencies\'.  This decorator does imports. E.g.:\n\n            @util.dependencies("sqlalchemy.sql.schema")\n\n    This hook scans for included SQLAlchemy modules and then scans those modules for any util.dependencies and marks\n    those modules as hidden imports.\n    '
    if not check_requirement('sqlalchemy >= 0.9'):
        return
    depend_regex = re.compile('@util.dependencies\\([\\\'"](.*?)[\\\'"]\\)')
    hidden_imports_set = set()
    known_imports = set()
    for node in hook_api.module_graph.iter_graph(start=hook_api.module):
        if isinstance(node, SourceModule) and node.identifier.startswith('sqlalchemy.'):
            known_imports.add(node.identifier)
            with open(node.filename, 'rb') as f:
                encoding = guess_encoding(f)
            with open(node.filename, 'r', encoding=encoding) as f:
                for match in depend_regex.findall(f.read()):
                    hidden_imports_set.add(match)
    hidden_imports_set -= known_imports
    if len(hidden_imports_set):
        logger.info('  Found %d sqlalchemy hidden imports', len(hidden_imports_set))
        hook_api.add_imports(*list(hidden_imports_set))