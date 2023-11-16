import pytest
import salt.fileserver.roots as roots
from salt.utils.odict import OrderedDict
from tests.support.mock import patch
pytestmark = [pytest.mark.windows_whitelisted]

@pytest.fixture(scope='function')
def configure_loader_modules(minion_opts):
    if False:
        i = 10
        return i + 15
    return {roots: {'__opts__': minion_opts}}

def test_symlink_list(state_tree):
    if False:
        print('Hello World!')
    with pytest.helpers.temp_file('target', 'data', state_tree) as target:
        link = state_tree / 'link'
        link.symlink_to(str(target))
        ret = roots.symlink_list({'saltenv': 'base'})
        assert ret == {'link': str(target)}

@pytest.mark.parametrize('env', ('base', 'something-else', 'cool_path_123', '__env__'))
def test_fileserver_roots_find_file_envs_path_substitution(env, minion_opts, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test fileserver access to a dynamic path using __env__\n    '
    fn = 'test.txt'
    if env == '__env__':
        actual_env = 'dynamic'
        leaf_dir = actual_env
    else:
        actual_env = env
        leaf_dir = '__env__'
    envpath = tmp_path / leaf_dir
    envpath.mkdir(parents=True, exist_ok=True)
    filepath = envpath / fn
    filepath.touch()
    expected = OrderedDict()
    expected['rel'] = fn
    expected['path'] = str(filepath)
    minion_opts['file_roots'] = OrderedDict()
    minion_opts['file_roots'][env] = [str(tmp_path / leaf_dir)]
    with patch('salt.fileserver.roots.__opts__', minion_opts, create=True):
        ret = roots.find_file(fn, saltenv=actual_env)
    ret.pop('stat')
    assert ret == expected

@pytest.mark.parametrize('saltenv', ('base', 'something-else', 'cool_path_123', '__env__'))
def test_fileserver_roots__file_lists_envs_path_substitution(saltenv, tmp_path, minion_opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test fileserver access to a dynamic path using __env__\n    '
    other_env = 'something_completely_different'
    other_filename = 'different.txt'
    expected_filename = 'test.txt'
    expected = [expected_filename]
    expected_different_ret = [other_filename]
    actual_env = 'dynamic' if saltenv == '__env__' else saltenv
    file_roots = tmp_path / '__env__' / 'cool'
    envpath = tmp_path / actual_env / 'cool'
    otherpath = tmp_path / other_env / 'cool'
    envpath.mkdir(parents=True, exist_ok=True)
    otherpath.mkdir(parents=True, exist_ok=True)
    (envpath / expected_filename).touch()
    (otherpath / other_filename).touch()
    minion_opts['file_roots'] = OrderedDict()
    minion_opts['file_roots']['__env__'] = [str(file_roots)]
    with patch('salt.fileserver.roots.__opts__', minion_opts, create=True):
        ret = roots._file_lists({'saltenv': actual_env}, 'files')
        different_ret = roots._file_lists({'saltenv': other_env}, 'files')
    assert ret == expected
    assert different_ret != ret
    assert different_ret == expected_different_ret