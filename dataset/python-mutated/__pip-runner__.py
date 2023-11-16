"""Execute exactly this copy of pip, within a different environment.

This file is named as it is, to ensure that this module can't be imported via
an import statement.
"""
import sys
PYTHON_REQUIRES = (3, 7)

def version_str(version):
    if False:
        return 10
    return '.'.join((str(v) for v in version))
if sys.version_info[:2] < PYTHON_REQUIRES:
    raise SystemExit('This version of pip does not support python {} (requires >={}).'.format(version_str(sys.version_info[:2]), version_str(PYTHON_REQUIRES)))
import runpy
from importlib.machinery import PathFinder
from os.path import dirname
PIP_SOURCES_ROOT = dirname(dirname(__file__))

class PipImportRedirectingFinder:

    @classmethod
    def find_spec(self, fullname, path=None, target=None):
        if False:
            for i in range(10):
                print('nop')
        if fullname != 'pip':
            return None
        spec = PathFinder.find_spec(fullname, [PIP_SOURCES_ROOT], target)
        assert spec, (PIP_SOURCES_ROOT, fullname)
        return spec
sys.meta_path.insert(0, PipImportRedirectingFinder())
assert __name__ == '__main__', 'Cannot run __pip-runner__.py as a non-main module'
runpy.run_module('pip', run_name='__main__', alter_sys=True)