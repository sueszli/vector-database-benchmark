from __future__ import annotations
import pytest
from ansible.modules.unarchive import ZipArchive, TgzArchive

@pytest.fixture
def fake_ansible_module():
    if False:
        print('Hello World!')
    return FakeAnsibleModule()

class FakeAnsibleModule:

    def __init__(self):
        if False:
            print('Hello World!')
        self.params = {}
        self.tmpdir = None

class TestCaseZipArchive:

    @pytest.mark.parametrize('side_effect, expected_reason', (([ValueError, '/bin/zipinfo'], "Unable to find required 'unzip'"), (ValueError, "Unable to find required 'unzip' or 'zipinfo'")))
    def test_no_zip_zipinfo_binary(self, mocker, fake_ansible_module, side_effect, expected_reason):
        if False:
            return 10
        mocker.patch('ansible.modules.unarchive.get_bin_path', side_effect=side_effect)
        fake_ansible_module.params = {'extra_opts': '', 'exclude': '', 'include': '', 'io_buffer_size': 65536}
        z = ZipArchive(src='', b_dest='', file_args='', module=fake_ansible_module)
        (can_handle, reason) = z.can_handle_archive()
        assert can_handle is False
        assert expected_reason in reason
        assert z.cmd_path is None

class TestCaseTgzArchive:

    def test_no_tar_binary(self, mocker, fake_ansible_module):
        if False:
            i = 10
            return i + 15
        mocker.patch('ansible.modules.unarchive.get_bin_path', side_effect=ValueError)
        fake_ansible_module.params = {'extra_opts': '', 'exclude': '', 'include': '', 'io_buffer_size': 65536}
        fake_ansible_module.check_mode = False
        t = TgzArchive(src='', b_dest='', file_args='', module=fake_ansible_module)
        (can_handle, reason) = t.can_handle_archive()
        assert can_handle is False
        assert 'Unable to find required' in reason
        assert t.cmd_path is None
        assert t.tar_type is None