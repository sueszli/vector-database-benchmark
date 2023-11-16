from __future__ import annotations
import errno
import json
import pytest
from unittest.mock import mock_open, patch
from ansible.module_utils import basic
from ansible.module_utils.common.text.converters import to_bytes
import builtins

@pytest.fixture
def no_args_module_exec():
    if False:
        for i in range(10):
            print('nop')
    with patch.object(basic, '_ANSIBLE_ARGS', b'{"ANSIBLE_MODULE_ARGS": {}}'):
        yield

def no_args_module(selinux_enabled=None, selinux_mls_enabled=None):
    if False:
        return 10
    am = basic.AnsibleModule(argument_spec={})
    if isinstance(selinux_enabled, bool):
        patch.object(am, 'selinux_enabled', return_value=selinux_enabled).start()
    if isinstance(selinux_mls_enabled, bool):
        patch.object(am, 'selinux_mls_enabled', return_value=selinux_mls_enabled).start()
    return am

@pytest.mark.usefixtures('no_args_module_exec')
class TestSELinuxMU:

    def test_selinux_enabled(self):
        if False:
            return 10
        with patch.object(basic, 'HAVE_SELINUX', False):
            assert no_args_module().selinux_enabled() is False
            disabled_mod = no_args_module()
            with patch.object(basic, 'selinux', create=True) as selinux:
                selinux.is_selinux_enabled.return_value = 0
                assert disabled_mod.selinux_enabled() is False
        assert disabled_mod.selinux_enabled() is False
        with patch.object(basic, 'HAVE_SELINUX', True):
            enabled_mod = no_args_module()
            with patch.object(basic, 'selinux', create=True) as selinux:
                selinux.is_selinux_enabled.return_value = 1
                assert enabled_mod.selinux_enabled() is True
        assert enabled_mod.selinux_enabled() is True

    def test_selinux_mls_enabled(self):
        if False:
            while True:
                i = 10
        with patch.object(basic, 'HAVE_SELINUX', False):
            assert no_args_module().selinux_mls_enabled() is False
            with patch.object(basic, 'selinux', create=True) as selinux:
                selinux.is_selinux_mls_enabled.return_value = 0
                assert no_args_module(selinux_enabled=False).selinux_mls_enabled() is False
        with patch.object(basic, 'HAVE_SELINUX', True):
            with patch.object(basic, 'selinux', create=True) as selinux:
                selinux.is_selinux_mls_enabled.return_value = 1
                assert no_args_module(selinux_enabled=True).selinux_mls_enabled() is True

    def test_selinux_initial_context(self):
        if False:
            for i in range(10):
                print('nop')
        assert no_args_module(selinux_enabled=False, selinux_mls_enabled=False).selinux_initial_context() == [None, None, None]
        assert no_args_module(selinux_enabled=True, selinux_mls_enabled=False).selinux_initial_context() == [None, None, None]
        assert no_args_module(selinux_enabled=True, selinux_mls_enabled=True).selinux_initial_context() == [None, None, None, None]

    def test_selinux_default_context(self):
        if False:
            i = 10
            return i + 15
        with patch.object(basic, 'HAVE_SELINUX', False):
            assert no_args_module().selinux_default_context(path='/foo/bar') == [None, None, None]
        am = no_args_module(selinux_enabled=True, selinux_mls_enabled=True)
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.matchpathcon.return_value = [0, 'unconfined_u:object_r:default_t:s0']
            assert am.selinux_default_context(path='/foo/bar') == ['unconfined_u', 'object_r', 'default_t', 's0']
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.matchpathcon.return_value = [-1, '']
            assert am.selinux_default_context(path='/foo/bar') == [None, None, None, None]
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.matchpathcon.side_effect = OSError
            assert am.selinux_default_context(path='/foo/bar') == [None, None, None, None]

    def test_selinux_context(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(basic, 'HAVE_SELINUX', False):
            assert no_args_module().selinux_context(path='/foo/bar') == [None, None, None]
        am = no_args_module(selinux_enabled=True, selinux_mls_enabled=True)
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lgetfilecon_raw.return_value = [0, 'unconfined_u:object_r:default_t:s0']
            assert am.selinux_context(path='/foo/bar') == ['unconfined_u', 'object_r', 'default_t', 's0']
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lgetfilecon_raw.return_value = [-1, '']
            assert am.selinux_context(path='/foo/bar') == [None, None, None, None]
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lgetfilecon_raw.side_effect = OSError(errno.ENOENT, 'NotFound')
            with pytest.raises(SystemExit):
                am.selinux_context(path='/foo/bar')
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lgetfilecon_raw.side_effect = OSError()
            with pytest.raises(SystemExit):
                am.selinux_context(path='/foo/bar')

    def test_is_special_selinux_path(self):
        if False:
            while True:
                i = 10
        args = to_bytes(json.dumps(dict(ANSIBLE_MODULE_ARGS={'_ansible_selinux_special_fs': 'nfs,nfsd,foos', '_ansible_remote_tmp': '/tmp', '_ansible_keep_remote_files': False})))
        with patch.object(basic, '_ANSIBLE_ARGS', args):
            am = basic.AnsibleModule(argument_spec=dict())

            def _mock_find_mount_point(path):
                if False:
                    for i in range(10):
                        print('nop')
                if path.startswith('/some/path'):
                    return '/some/path'
                elif path.startswith('/weird/random/fstype'):
                    return '/weird/random/fstype'
                return '/'
            am.find_mount_point = _mock_find_mount_point
            am.selinux_context = lambda path: ['foo_u', 'foo_r', 'foo_t', 's0']
            m = mock_open()
            m.side_effect = OSError
            with patch.object(builtins, 'open', m, create=True):
                assert am.is_special_selinux_path('/some/path/that/should/be/nfs') == (False, None)
            mount_data = ['/dev/disk1 / ext4 rw,seclabel,relatime,data=ordered 0 0\n', '10.1.1.1:/path/to/nfs /some/path nfs ro 0 0\n', 'whatever /weird/random/fstype foos rw 0 0\n']
            m = mock_open(read_data=''.join(mount_data))
            m.return_value.readlines.return_value = mount_data
            with patch.object(builtins, 'open', m, create=True):
                assert am.is_special_selinux_path('/some/random/path') == (False, None)
                assert am.is_special_selinux_path('/some/path/that/should/be/nfs') == (True, ['foo_u', 'foo_r', 'foo_t', 's0'])
                assert am.is_special_selinux_path('/weird/random/fstype/path') == (True, ['foo_u', 'foo_r', 'foo_t', 's0'])

    def test_set_context_if_different(self):
        if False:
            for i in range(10):
                print('nop')
        am = no_args_module(selinux_enabled=False)
        assert am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], True) is True
        assert am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], False) is False
        am = no_args_module(selinux_enabled=True, selinux_mls_enabled=True)
        am.selinux_context = lambda path: ['bar_u', 'bar_r', None, None]
        am.is_special_selinux_path = lambda path: (False, None)
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lsetfilecon.return_value = 0
            assert am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], False) is True
            selinux.lsetfilecon.assert_called_with('/path/to/file', 'foo_u:foo_r:foo_t:s0')
            selinux.lsetfilecon.reset_mock()
            am.check_mode = True
            assert am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], False) is True
            assert not selinux.lsetfilecon.called
            am.check_mode = False
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lsetfilecon.return_value = 1
            with pytest.raises(SystemExit):
                am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], True)
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lsetfilecon.side_effect = OSError
            with pytest.raises(SystemExit):
                am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], True)
        am.is_special_selinux_path = lambda path: (True, ['sp_u', 'sp_r', 'sp_t', 's0'])
        with patch.object(basic, 'selinux', create=True) as selinux:
            selinux.lsetfilecon.return_value = 0
            assert am.set_context_if_different('/path/to/file', ['foo_u', 'foo_r', 'foo_t', 's0'], False) is True
            selinux.lsetfilecon.assert_called_with('/path/to/file', 'sp_u:sp_r:sp_t:s0')