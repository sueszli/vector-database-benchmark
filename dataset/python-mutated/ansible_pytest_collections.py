"""Enable unit testing of Ansible collections. PYTEST_DONT_REWRITE"""
from __future__ import annotations
import os
ANSIBLE_COLLECTIONS_PATH = os.path.join(os.environ['ANSIBLE_COLLECTIONS_PATH'], 'ansible_collections')
ANSIBLE_CONTROLLER_MIN_PYTHON_VERSION = tuple((int(x) for x in os.environ['ANSIBLE_CONTROLLER_MIN_PYTHON_VERSION'].split('.')))

def collection_resolve_package_path(path):
    if False:
        print('Hello World!')
    'Configure the Python package path so that pytest can find our collections.'
    for parent in path.parents:
        if str(parent) == ANSIBLE_COLLECTIONS_PATH:
            return parent
    raise Exception('File "%s" not found in collection path "%s".' % (path, ANSIBLE_COLLECTIONS_PATH))

def collection_pypkgpath(self):
    if False:
        while True:
            i = 10
    'Configure the Python package path so that pytest can find our collections.'
    for parent in self.parts(reverse=True):
        if str(parent) == ANSIBLE_COLLECTIONS_PATH:
            return parent
    raise Exception('File "%s" not found in collection path "%s".' % (self.strpath, ANSIBLE_COLLECTIONS_PATH))

def enable_assertion_rewriting_hook():
    if False:
        return 10
    "\n    Enable pytest's AssertionRewritingHook on Python 3.x.\n    This is necessary because the Ansible collection loader intercepts imports before the pytest provided loader ever sees them.\n    "
    import sys
    hook_name = '_pytest.assertion.rewrite.AssertionRewritingHook'
    hooks = [hook for hook in sys.meta_path if hook.__class__.__module__ + '.' + hook.__class__.__qualname__ == hook_name]
    if len(hooks) != 1:
        raise Exception('Found {} instance(s) of "{}" in sys.meta_path.'.format(len(hooks), hook_name))
    assertion_rewriting_hook = hooks[0]

    def exec_module(self, module):
        if False:
            for i in range(10):
                print('nop')
        if self._redirect_module:
            return
        code_obj = self.get_code(self._fullname)
        if code_obj is not None:
            should_rewrite = self._package_to_load == 'conftest' or self._package_to_load.startswith('test_')
            if should_rewrite:
                assertion_rewriting_hook.exec_module(module)
            else:
                exec(code_obj, module.__dict__)
    from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionPkgLoaderBase
    _AnsibleCollectionPkgLoaderBase.exec_module = exec_module

def pytest_configure():
    if False:
        print('Hello World!')
    'Configure this pytest plugin.'
    try:
        if pytest_configure.executed:
            return
    except AttributeError:
        pytest_configure.executed = True
    enable_assertion_rewriting_hook()
    from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder
    _AnsibleCollectionFinder(paths=[os.path.dirname(ANSIBLE_COLLECTIONS_PATH)])._install()
    try:
        from _pytest import pathlib as _pytest_pathlib
    except ImportError:
        _pytest_pathlib = None
    if hasattr(_pytest_pathlib, 'resolve_package_path'):
        _pytest_pathlib.resolve_package_path = collection_resolve_package_path
    else:
        import py._path.local
        py._path.local.LocalPath.pypkgpath = collection_pypkgpath
pytest_configure()