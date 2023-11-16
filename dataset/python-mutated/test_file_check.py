import getpass
import logging
import os
import pytest
import salt.modules.file as filemod
import salt.utils.files
import salt.utils.platform
from tests.support.mock import Mock, patch
log = logging.getLogger(__name__)

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {filemod: {'__context__': {}, '__opts__': {'test': False}}}

@pytest.fixture
def tfile(tmp_path):
    if False:
        while True:
            i = 10
    filename = str(tmp_path / 'file-check-test-file')
    with salt.utils.files.fopen(filename, 'w') as fp:
        fp.write('Hi hello! I am a file.')
    os.chmod(filename, 420)
    yield filename
    os.remove(filename)

@pytest.fixture
def a_link(tmp_path, tfile):
    if False:
        print('Hello World!')
    linkname = str(tmp_path / 'a_link')
    os.symlink(tfile, linkname)
    yield linkname
    os.remove(linkname)

def get_link_perms():
    if False:
        return 10
    if salt.utils.platform.is_linux():
        return '0777'
    return '0755'

@pytest.mark.skip_on_windows(reason='os.symlink is not available on Windows')
def test_check_file_meta_follow_symlinks(a_link, tfile):
    if False:
        while True:
            i = 10
    user = getpass.getuser()
    lperms = get_link_perms()
    ret = filemod.check_file_meta(a_link, tfile, None, None, user, None, lperms, None, None)
    assert ret == {}
    ret = filemod.check_file_meta(a_link, tfile, None, None, user, None, '0644', None, None)
    assert ret == {'mode': '0644'}
    ret = filemod.check_file_meta(a_link, tfile, None, None, user, None, '0644', None, None, follow_symlinks=True)
    assert ret == {}

@pytest.mark.skip_on_windows(reason='os.symlink is not available on Windows')
def test_check_managed_follow_symlinks(a_link, tfile):
    if False:
        while True:
            i = 10
    user = getpass.getuser()
    lperms = get_link_perms()
    a_link = '/' + a_link
    (ret, comments) = filemod.check_managed(a_link, tfile, None, None, user, None, lperms, None, None, None, None, None)
    assert ret is True
    assert comments == 'The file {} is in the correct state'.format(a_link)
    (ret, comments) = filemod.check_managed(a_link, tfile, None, None, user, None, '0644', None, None, None, None, None)
    assert ret is None
    assert comments == 'The following values are set to be changed:\nmode: 0644\n'
    (ret, comments) = filemod.check_managed(a_link, tfile, None, None, user, None, '0644', None, None, None, None, None, follow_symlinks=True)
    assert ret is True
    assert comments == 'The file {} is in the correct state'.format(a_link)

@pytest.mark.skip_on_windows(reason='os.symlink is not available on Windows')
def test_check_managed_changes_follow_symlinks(a_link, tfile):
    if False:
        i = 10
        return i + 15
    user = getpass.getuser()
    lperms = get_link_perms()
    ret = filemod.check_managed_changes(a_link, tfile, None, None, user, None, lperms, None, None, None, None, None)
    assert ret == {}
    ret = filemod.check_managed_changes(a_link, tfile, None, None, user, None, '0644', None, None, None, None, None)
    assert ret == {'mode': '0644'}
    ret = filemod.check_managed_changes(a_link, tfile, None, None, user, None, '0644', None, None, None, None, None, follow_symlinks=True)
    assert ret == {}

@pytest.mark.skip_on_windows(reason='os.symlink is not available on Windows')
@pytest.mark.parametrize('input,expected', [({'user': 'cuser', 'group': 'cgroup'}, {'user': 'cuser', 'group': 'cgroup'}), ({'user': 'luser', 'group': 'lgroup'}, {}), ({'user': 1001, 'group': 2001}, {'user': 1001, 'group': 2001}), ({'user': 3001, 'group': 4001}, {})])
def test_check_perms_user_group_name_and_id(input, expected):
    if False:
        i = 10
        return i + 15
    filename = '/path/to/fnord'
    with patch('os.path.exists', Mock(return_value=True)):
        stat_out = {'user': 'luser', 'group': 'lgroup', 'uid': 3001, 'gid': 4001, 'mode': '123'}
        patch_stats = patch('salt.modules.file.stats', Mock(return_value=stat_out))

        def fake_chown(cmd, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            for (k, v) in input.items():
                stat_out.update({k: v})
        patch_chown = patch('salt.modules.file.chown', Mock(side_effect=fake_chown))
        with patch_stats, patch_chown:
            (ret, pre_post) = filemod.check_perms(name=filename, ret={}, user=input['user'], group=input['group'], mode='123', follow_symlinks=False)
            assert ret['changes'] == expected