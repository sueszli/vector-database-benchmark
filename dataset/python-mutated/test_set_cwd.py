from __future__ import annotations
import json
import os
import tempfile
from unittest.mock import patch
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils import basic

class TestAnsibleModuleSetCwd:

    def test_set_cwd(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        'make sure /tmp is used'

        def mock_getcwd():
            if False:
                while True:
                    i = 10
            return '/tmp'

        def mock_access(path, perm):
            if False:
                return 10
            return True

        def mock_chdir(path):
            if False:
                print('Hello World!')
            pass
        monkeypatch.setattr(os, 'getcwd', mock_getcwd)
        monkeypatch.setattr(os, 'access', mock_access)
        monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
        with patch('time.time', return_value=42):
            am = basic.AnsibleModule(argument_spec={})
        result = am._set_cwd()
        assert result == '/tmp'

    def test_set_cwd_unreadable_use_self_tmpdir(self, monkeypatch):
        if False:
            return 10
        "pwd is not readable, use instance's tmpdir property"

        def mock_getcwd():
            if False:
                print('Hello World!')
            return '/tmp'

        def mock_access(path, perm):
            if False:
                return 10
            if path == '/tmp' and perm == 4:
                return False
            return True

        def mock_expandvars(var):
            if False:
                for i in range(10):
                    print('nop')
            if var == '$HOME':
                return '/home/foobar'
            return var

        def mock_gettempdir():
            if False:
                while True:
                    i = 10
            return '/tmp/testdir'

        def mock_chdir(path):
            if False:
                i = 10
                return i + 15
            if path == '/tmp':
                raise Exception()
            return
        monkeypatch.setattr(os, 'getcwd', mock_getcwd)
        monkeypatch.setattr(os, 'chdir', mock_chdir)
        monkeypatch.setattr(os, 'access', mock_access)
        monkeypatch.setattr(os.path, 'expandvars', mock_expandvars)
        monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
        with patch('time.time', return_value=42):
            am = basic.AnsibleModule(argument_spec={})
        am._tmpdir = '/tmp2'
        result = am._set_cwd()
        assert result == am._tmpdir

    def test_set_cwd_unreadable_use_home(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        'cwd and instance tmpdir are unreadable, use home'

        def mock_getcwd():
            if False:
                for i in range(10):
                    print('nop')
            return '/tmp'

        def mock_access(path, perm):
            if False:
                return 10
            if path in ['/tmp', '/tmp2'] and perm == 4:
                return False
            return True

        def mock_expandvars(var):
            if False:
                print('Hello World!')
            if var == '$HOME':
                return '/home/foobar'
            return var

        def mock_gettempdir():
            if False:
                return 10
            return '/tmp/testdir'

        def mock_chdir(path):
            if False:
                return 10
            if path == '/tmp':
                raise Exception()
            return
        monkeypatch.setattr(os, 'getcwd', mock_getcwd)
        monkeypatch.setattr(os, 'chdir', mock_chdir)
        monkeypatch.setattr(os, 'access', mock_access)
        monkeypatch.setattr(os.path, 'expandvars', mock_expandvars)
        monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
        with patch('time.time', return_value=42):
            am = basic.AnsibleModule(argument_spec={})
        am._tmpdir = '/tmp2'
        result = am._set_cwd()
        assert result == '/home/foobar'

    def test_set_cwd_unreadable_use_gettempdir(self, monkeypatch):
        if False:
            print('Hello World!')
        'fallback to tempfile.gettempdir'
        thisdir = None

        def mock_getcwd():
            if False:
                while True:
                    i = 10
            return '/tmp'

        def mock_access(path, perm):
            if False:
                for i in range(10):
                    print('nop')
            if path in ['/tmp', '/tmp2', '/home/foobar'] and perm == 4:
                return False
            return True

        def mock_expandvars(var):
            if False:
                print('Hello World!')
            if var == '$HOME':
                return '/home/foobar'
            return var

        def mock_gettempdir():
            if False:
                for i in range(10):
                    print('nop')
            return '/tmp3'

        def mock_chdir(path):
            if False:
                for i in range(10):
                    print('nop')
            if path == '/tmp':
                raise Exception()
            thisdir = path
        monkeypatch.setattr(os, 'getcwd', mock_getcwd)
        monkeypatch.setattr(os, 'chdir', mock_chdir)
        monkeypatch.setattr(os, 'access', mock_access)
        monkeypatch.setattr(os.path, 'expandvars', mock_expandvars)
        monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
        with patch('time.time', return_value=42):
            am = basic.AnsibleModule(argument_spec={})
        am._tmpdir = '/tmp2'
        monkeypatch.setattr(tempfile, 'gettempdir', mock_gettempdir)
        result = am._set_cwd()
        assert result == '/tmp3'

    def test_set_cwd_unreadable_use_None(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        'all paths are unreable, should return None and not an exception'

        def mock_getcwd():
            if False:
                i = 10
                return i + 15
            return '/tmp'

        def mock_access(path, perm):
            if False:
                while True:
                    i = 10
            if path in ['/tmp', '/tmp2', '/tmp3', '/home/foobar'] and perm == 4:
                return False
            return True

        def mock_expandvars(var):
            if False:
                i = 10
                return i + 15
            if var == '$HOME':
                return '/home/foobar'
            return var

        def mock_gettempdir():
            if False:
                for i in range(10):
                    print('nop')
            return '/tmp3'

        def mock_chdir(path):
            if False:
                while True:
                    i = 10
            if path == '/tmp':
                raise Exception()
        monkeypatch.setattr(os, 'getcwd', mock_getcwd)
        monkeypatch.setattr(os, 'chdir', mock_chdir)
        monkeypatch.setattr(os, 'access', mock_access)
        monkeypatch.setattr(os.path, 'expandvars', mock_expandvars)
        monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
        with patch('time.time', return_value=42):
            am = basic.AnsibleModule(argument_spec={})
        am._tmpdir = '/tmp2'
        monkeypatch.setattr(tempfile, 'gettempdir', mock_gettempdir)
        result = am._set_cwd()
        assert result is None