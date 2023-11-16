from __future__ import annotations
import os
import pytest
import zipfile
from collections import namedtuple
from io import BytesIO
import ansible.errors
from ansible.executor.module_common import recursive_finder
from ansible.plugins.loader import init_plugin_loader
MODULE_UTILS_BASIC_FILES = frozenset(('ansible/__init__.py', 'ansible/module_utils/__init__.py', 'ansible/module_utils/_text.py', 'ansible/module_utils/basic.py', 'ansible/module_utils/six/__init__.py', 'ansible/module_utils/_text.py', 'ansible/module_utils/common/collections.py', 'ansible/module_utils/common/parameters.py', 'ansible/module_utils/common/warnings.py', 'ansible/module_utils/parsing/convert_bool.py', 'ansible/module_utils/common/__init__.py', 'ansible/module_utils/common/file.py', 'ansible/module_utils/common/locale.py', 'ansible/module_utils/common/process.py', 'ansible/module_utils/common/sys_info.py', 'ansible/module_utils/common/text/__init__.py', 'ansible/module_utils/common/text/converters.py', 'ansible/module_utils/common/text/formatters.py', 'ansible/module_utils/common/validation.py', 'ansible/module_utils/common/_utils.py', 'ansible/module_utils/common/arg_spec.py', 'ansible/module_utils/compat/__init__.py', 'ansible/module_utils/compat/selinux.py', 'ansible/module_utils/distro/__init__.py', 'ansible/module_utils/distro/_distro.py', 'ansible/module_utils/errors.py', 'ansible/module_utils/parsing/__init__.py', 'ansible/module_utils/parsing/convert_bool.py', 'ansible/module_utils/pycompat24.py', 'ansible/module_utils/six/__init__.py'))
ONLY_BASIC_FILE = frozenset(('ansible/module_utils/basic.py',))
ANSIBLE_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'lib', 'ansible')

@pytest.fixture
def finder_containers():
    if False:
        i = 10
        return i + 15
    init_plugin_loader()
    FinderContainers = namedtuple('FinderContainers', ['zf'])
    zipoutput = BytesIO()
    zf = zipfile.ZipFile(zipoutput, mode='w', compression=zipfile.ZIP_STORED)
    return FinderContainers(zf)

class TestRecursiveFinder(object):

    def test_no_module_utils(self, finder_containers):
        if False:
            return 10
        name = 'ping'
        data = b'#!/usr/bin/python\nreturn \'{"changed": false}\''
        recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'ping.py'), data, *finder_containers)
        assert frozenset(finder_containers.zf.namelist()) == MODULE_UTILS_BASIC_FILES

    def test_module_utils_with_syntax_error(self, finder_containers):
        if False:
            print('Hello World!')
        name = 'fake_module'
        data = b'#!/usr/bin/python\ndef something(:\n   pass\n'
        with pytest.raises(ansible.errors.AnsibleError) as exec_info:
            recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'fake_module.py'), data, *finder_containers)
        assert 'Unable to import fake_module due to invalid syntax' in str(exec_info.value)

    def test_module_utils_with_identation_error(self, finder_containers):
        if False:
            print('Hello World!')
        name = 'fake_module'
        data = b'#!/usr/bin/python\n    def something():\n    pass\n'
        with pytest.raises(ansible.errors.AnsibleError) as exec_info:
            recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'fake_module.py'), data, *finder_containers)
        assert 'Unable to import fake_module due to unexpected indent' in str(exec_info.value)

    def test_from_import_six(self, finder_containers):
        if False:
            return 10
        name = 'ping'
        data = b'#!/usr/bin/python\nfrom ansible.module_utils import six'
        recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'ping.py'), data, *finder_containers)
        assert frozenset(finder_containers.zf.namelist()) == frozenset(('ansible/module_utils/six/__init__.py',)).union(MODULE_UTILS_BASIC_FILES)

    def test_import_six(self, finder_containers):
        if False:
            while True:
                i = 10
        name = 'ping'
        data = b'#!/usr/bin/python\nimport ansible.module_utils.six'
        recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'ping.py'), data, *finder_containers)
        assert frozenset(finder_containers.zf.namelist()) == frozenset(('ansible/module_utils/six/__init__.py',)).union(MODULE_UTILS_BASIC_FILES)

    def test_import_six_from_many_submodules(self, finder_containers):
        if False:
            while True:
                i = 10
        name = 'ping'
        data = b'#!/usr/bin/python\nfrom ansible.module_utils.six.moves.urllib.parse import urlparse'
        recursive_finder(name, os.path.join(ANSIBLE_LIB, 'modules', 'system', 'ping.py'), data, *finder_containers)
        assert frozenset(finder_containers.zf.namelist()) == frozenset(('ansible/module_utils/six/__init__.py',)).union(MODULE_UTILS_BASIC_FILES)