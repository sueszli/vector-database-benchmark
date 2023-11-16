import textwrap
import pytest
import tornado.ioloop
import salt.fileserver.gitfs as gitfs
import salt.utils.files
import salt.utils.gitfs
import salt.utils.platform
import salt.utils.win_functions
import salt.utils.yaml
from salt.utils.gitfs import GITPYTHON_MINVER, GITPYTHON_VERSION
from tests.support.mock import patch
try:
    import git
    HAS_GITPYTHON = GITPYTHON_VERSION >= GITPYTHON_MINVER
except (ImportError, AttributeError):
    HAS_GITPYTHON = False
pytestmark = [pytest.mark.skipif(not HAS_GITPYTHON, reason='GitPython >= {} required'.format(GITPYTHON_MINVER))]

@pytest.fixture
def configure_loader_modules(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    opts = {'sock_dir': str(tmp_path / 'sock_dir'), 'gitfs_remotes': ['file://' + str(tmp_path / 'repo_dir')], 'cachedir': str(tmp_path / 'cache_dir'), 'gitfs_root': '', 'fileserver_backend': ['gitfs'], 'gitfs_base': 'master', 'gitfs_fallback': '', 'fileserver_events': True, 'transport': 'zeromq', 'gitfs_mountpoint': '', 'gitfs_saltenv': [], 'gitfs_saltenv_whitelist': [], 'gitfs_saltenv_blacklist': [], 'gitfs_user': '', 'gitfs_password': '', 'gitfs_insecure_auth': False, 'gitfs_privkey': '', 'gitfs_pubkey': '', 'gitfs_passphrase': '', 'gitfs_refspecs': ['+refs/heads/*:refs/remotes/origin/*', '+refs/tags/*:refs/tags/*'], 'gitfs_ssl_verify': True, 'gitfs_disable_saltenv_mapping': False, 'gitfs_ref_types': ['branch', 'tag', 'sha'], 'gitfs_update_interval': 60, '__role': 'master'}
    if salt.utils.platform.is_windows():
        opts['gitfs_remotes'][0] = opts['gitfs_remotes'][0].replace('\\', '/')
    return {gitfs: {'__opts__': opts}}

@pytest.fixture(scope='module', autouse=True)
def clear_instance_map():
    if False:
        for i in range(10):
            print('nop')
    try:
        del salt.utils.gitfs.GitFS.instance_map[tornado.ioloop.IOLoop.current()]
    except KeyError:
        pass

def test_per_saltenv_config():
    if False:
        return 10
    opts_override = textwrap.dedent('\n        gitfs_root: salt\n\n        gitfs_saltenv:\n          - baz:\n            # when loaded, the "salt://" prefix will be removed\n            - mountpoint: salt://baz_mountpoint\n            - ref: baz_branch\n            - root: baz_root\n\n        gitfs_remotes:\n\n          - file://{0}tmp/repo1:\n            - saltenv:\n              - foo:\n                - ref: foo_branch\n                - root: foo_root\n\n          - file://{0}tmp/repo2:\n            - mountpoint: repo2\n            - saltenv:\n              - baz:\n                - mountpoint: abc\n    '.format('/' if salt.utils.platform.is_windows() else ''))
    with patch.dict(gitfs.__opts__, salt.utils.yaml.safe_load(opts_override)):
        git_fs = salt.utils.gitfs.GitFS(gitfs.__opts__, gitfs.__opts__['gitfs_remotes'], per_remote_overrides=gitfs.PER_REMOTE_OVERRIDES, per_remote_only=gitfs.PER_REMOTE_ONLY)
    assert git_fs.remotes[0].mountpoint('foo') == ''
    assert git_fs.remotes[0].ref('foo') == 'foo_branch'
    assert git_fs.remotes[0].root('foo') == 'foo_root'
    assert git_fs.remotes[0].mountpoint('bar') == ''
    assert git_fs.remotes[0].ref('bar') == 'bar'
    assert git_fs.remotes[0].root('bar') == 'salt'
    assert git_fs.remotes[0].mountpoint('baz') == 'baz_mountpoint'
    assert git_fs.remotes[0].ref('baz') == 'baz_branch'
    assert git_fs.remotes[0].root('baz') == 'baz_root'
    assert git_fs.remotes[1].mountpoint('foo') == 'repo2'
    assert git_fs.remotes[1].ref('foo') == 'foo'
    assert git_fs.remotes[1].root('foo') == 'salt'
    assert git_fs.remotes[1].mountpoint('bar') == 'repo2'
    assert git_fs.remotes[1].ref('bar') == 'bar'
    assert git_fs.remotes[1].root('bar') == 'salt'
    assert git_fs.remotes[1].mountpoint('baz') == 'abc'
    assert git_fs.remotes[1].ref('baz') == 'baz_branch'
    assert git_fs.remotes[1].root('baz') == 'baz_root'