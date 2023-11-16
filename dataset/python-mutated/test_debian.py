import glob
import logging
import os
import pathlib
import shutil
import sys
from sysconfig import get_path
import _pytest._version
import attr
import pytest
import salt.modules.aptpkg
import salt.utils.files
from tests.conftest import CODE_DIR
from tests.support.mock import MagicMock, patch
PYTEST_GE_7 = getattr(_pytest._version, 'version_tuple', (-1, -1)) >= (7, 0)
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.destructive_test, pytest.mark.skip_if_not_root, pytest.mark.slow_test]

@pytest.fixture
def pkgrepo(states, grains):
    if False:
        while True:
            i = 10
    if grains['os_family'] != 'Debian':
        exc_kwargs = {}
        if PYTEST_GE_7:
            exc_kwargs['_use_item_location'] = True
        raise pytest.skip.Exception('Test only for debian based platforms', **exc_kwargs)
    return states.pkgrepo

@pytest.mark.requires_salt_states('pkgrepo.managed')
def test_adding_repo_file(pkgrepo, repo_uri, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    test adding a repo file using pkgrepo.managed\n    '
    repo_file = str(tmp_path / 'stable-binary.list')
    repo_content = f'deb {repo_uri} stable main'
    ret = pkgrepo.managed(name=repo_content, file=repo_file, clean_file=True)
    with salt.utils.files.fopen(repo_file, 'r') as fp:
        file_content = fp.read()
    assert file_content.strip() == repo_content

@pytest.mark.requires_salt_states('pkgrepo.managed')
def test_adding_repo_file_arch(pkgrepo, repo_uri, tmp_path, subtests):
    if False:
        return 10
    '\n    test adding a repo file using pkgrepo.managed\n    and setting architecture\n    '
    repo_file = str(tmp_path / 'stable-binary.list')
    repo_content = f'deb [arch=amd64  ] {repo_uri} stable main'
    pkgrepo.managed(name=repo_content, file=repo_file, clean_file=True)
    with salt.utils.files.fopen(repo_file, 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == f'deb [arch=amd64] {repo_uri} stable main'
    with subtests.test('With multiple archs'):
        repo_content = f'deb [arch=amd64,i386  ] {repo_uri} stable main'
        pkgrepo.managed(name=repo_content, file=repo_file, clean_file=True)
        with salt.utils.files.fopen(repo_file, 'r') as fp:
            file_content = fp.read()
            assert file_content.strip() == f'deb [arch=amd64,i386] {repo_uri} stable main'

@pytest.mark.requires_salt_states('pkgrepo.managed')
def test_adding_repo_file_cdrom(pkgrepo, tmp_path):
    if False:
        i = 10
        return i + 15
    "\n    test adding a repo file using pkgrepo.managed\n    The issue is that CDROM installs often have [] in the line, and we\n    should still add the repo even though it's not setting arch(for example)\n    "
    repo_file = str(tmp_path / 'cdrom.list')
    repo_content = 'deb cdrom:[Debian GNU/Linux 11.4.0 _Bullseye_ - Official amd64 NETINST 20220709-10:31]/ stable main'
    pkgrepo.managed(name=repo_content, file=repo_file, clean_file=True)
    with salt.utils.files.fopen(repo_file, 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == 'deb cdrom:[Debian GNU/Linux 11.4.0 _Bullseye_ - Official amd64 NETINST 20220709-10:31]/ stable main'

def system_aptsources_ids(value):
    if False:
        for i in range(10):
            print('nop')
    return f'{value.title()}(aptsources.sourceslist)'

@pytest.fixture(params=('with', 'without'), ids=system_aptsources_ids, scope='module', autouse=True)
def system_aptsources(request, grains):
    if False:
        return 10
    sys_modules = list(sys.modules)
    copied_paths = []
    exc_kwargs = {}
    if PYTEST_GE_7:
        exc_kwargs['_use_item_location'] = True
    if grains['os_family'] != 'Debian':
        raise pytest.skip.Exception('Test only for debian based platforms', **exc_kwargs)
    try:
        try:
            from aptsources import sourceslist
            if request.param == 'without':
                raise pytest.skip.Exception("This test is meant to run without the system aptsources package, but it's available from '{}'.".format(sourceslist.__file__), **exc_kwargs)
            else:
                yield request.param
        except ImportError:
            if request.param == 'without':
                yield
            else:
                copied_paths = []
                py_version_keys = ['{}'.format(*sys.version_info), '{}.{}'.format(*sys.version_info)]
                session_site_packages_dir = get_path('purelib')
                session_site_packages_dir = os.path.relpath(session_site_packages_dir, str(CODE_DIR))
                for py_version in py_version_keys:
                    dist_packages_path = '/usr/lib/python{}/dist-packages'.format(py_version)
                    if not os.path.isdir(dist_packages_path):
                        continue
                    for aptpkg in glob.glob(os.path.join(dist_packages_path, '*apt*')):
                        src = os.path.realpath(aptpkg)
                        dst = os.path.join(session_site_packages_dir, os.path.basename(src))
                        if os.path.exists(dst):
                            log.info('Not overwritting already existing %s with %s', dst, src)
                            continue
                        log.info('Copying %s into %s', src, dst)
                        copied_paths.append(dst)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copyfile(src, dst)
                if not copied_paths:
                    raise pytest.skip.Exception('aptsources.sourceslist python module not found', **exc_kwargs)
                yield request.param
    finally:
        for path in copied_paths:
            log.info('Deleting %r', path)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.unlink(path)
        for name in list(sys.modules):
            if name in sys_modules:
                continue
            if 'apt' not in name:
                continue
            log.debug("Removing '%s' from 'sys.modules'", name)
            sys.modules.pop(name)

@pytest.fixture
def ubuntu_state_tree(system_aptsources, state_tree, grains):
    if False:
        return 10
    if grains['os'] != 'Ubuntu':
        pytest.skip("Test only applicable to Ubuntu, not '{}'".format(grains['osfinger']))
    managed_sls_contents = "\n    {% set codename = grains['oscodename'] %}\n    {% set ubuntu_repos = [] %}\n    {% set beta = grains['oscodename'] in ['xenial', 'bionic', 'eoan', 'focal', 'groovy'] %}\n    {% set backports = grains['oscodename'] in ['xenial', 'bionic', 'eoan', 'focal'] %}\n\n    {%- if beta %}{%- do ubuntu_repos.append('firefox-beta') %}\n    firefox-beta:\n      pkgrepo.managed:\n        - name: deb http://ppa.launchpad.net/mozillateam/firefox-next/ubuntu {{ codename }} main\n        - dist: {{ codename }}\n        - file: /etc/apt/sources.list.d/firefox-beta.list\n        - keyid: CE49EC21\n        - keyserver: keyserver.ubuntu.com\n    {%- endif %}\n\n    {%- if backports %}{%- do ubuntu_repos.append('kubuntu-ppa') %}\n    kubuntu-ppa:\n      pkgrepo.managed:\n        - ppa: kubuntu-ppa/backports\n    {%- endif %}\n\n    pkgrepo-deps:\n      pkg.installed:\n        - pkgs:\n          - python3-apt\n          - software-properties-common\n    {%- for repo in ubuntu_repos -%}\n    {% if loop.first %}\n        - require_in:{%- endif %}\n          - pkgrepo: {{ repo }}\n    {%- endfor %}\n    "
    absent_sls_contents = "\n    firefox-beta:\n      pkgrepo.absent:\n        - name: deb http://ppa.launchpad.net/mozillateam/firefox-next/ubuntu {{ grains['oscodename'] }} main\n\n    kubuntu-ppa:\n      pkgrepo.absent:\n        - ppa: kubuntu-ppa/backports\n    "
    managed_state_file = pytest.helpers.temp_file('pkgrepo/managed.sls', managed_sls_contents, state_tree)
    absent_state_file = pytest.helpers.temp_file('pkgrepo/absent.sls', absent_sls_contents, state_tree)
    try:
        with managed_state_file, absent_state_file:
            yield
    finally:
        for pathstr in ('/etc/apt/sources.list.d/firefox-beta.list',):
            path = pathlib.Path(pathstr)
            if path.exists():
                path.unlink()

@pytest.mark.requires_salt_states('pkgrepo.managed', 'pkgrepo.absent')
def test_pkgrepo_managed_absent(modules, ubuntu_state_tree, subtests):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test adding a repo with the system aptsources package\n    '
    add_repo_test_passed = False
    with subtests.test('Add Repo'):
        ret = modules.state.sls('pkgrepo.managed')
        assert ret.failed is False
        for state in ret:
            assert state.result is True
        add_repo_test_passed = True
    with subtests.test('Remove Repo'):
        if add_repo_test_passed is False:
            pytest.skip('Adding the repo failed. Skipping.')
        ret = modules.state.sls('pkgrepo.absent')
        assert ret.failed is False
        for state in ret:
            assert state.result is True

@pytest.fixture
def multiple_comps_repo_file_caconical(grains):
    if False:
        return 10
    if grains['os'] != 'Ubuntu':
        pytest.skip("Test only applicable to Ubuntu, not '{}'".format(grains['osfinger']))
    repo_file_path = '/etc/apt/sources.list.d/99-salt-canonical-ubuntu.list'
    try:
        yield repo_file_path
    finally:
        try:
            os.unlink(repo_file_path)
        except OSError:
            pass

@pytest.fixture
def multiple_comps_repo_file_backports(grains):
    if False:
        while True:
            i = 10
    if grains['os'] != 'Ubuntu':
        pytest.skip("Test only applicable to Ubuntu, not '{}'".format(grains['osfinger']))
    repo_file_path = '/etc/apt/sources.list.d/99-salt-archive-ubuntu-focal-backports.list'
    try:
        yield repo_file_path
    finally:
        try:
            os.unlink(repo_file_path)
        except OSError:
            pass

@pytest.fixture
def multiple_comps_state_tree(multiple_comps_repo_file_caconical, multiple_comps_repo_file_backports, state_tree):
    if False:
        return 10
    sls_contents = "\n    ubuntu-backports:\n      pkgrepo.managed:\n        - name: 'deb http://fi.archive.ubuntu.com/ubuntu focal-backports'\n        - comps: main, restricted, universe, multiverse\n        - refresh: false\n        - disabled: false\n        - clean_file: true\n        - file: {}\n        - require_in:\n          - pkgrepo: canonical-ubuntu\n\n    canonical-ubuntu:\n      pkgrepo.managed:\n        - name: 'deb http://archive.canonical.com/ubuntu {{{{ salt['grains.get']('oscodename') }}}}'\n        - comps: partner\n        - refresh: false\n        - disabled: false\n        - clean_file: true\n        - file: {}\n    ".format(multiple_comps_repo_file_backports, multiple_comps_repo_file_caconical)
    with pytest.helpers.temp_file('multiple-comps-repos.sls', sls_contents, state_tree):
        yield

def test_managed_multiple_comps(modules, multiple_comps_state_tree):
    if False:
        i = 10
        return i + 15
    ret = modules.state.sls('multiple-comps-repos')
    assert ret.failed is False
    for state in ret:
        assert state.result is True
        assert state.changes
    ret = modules.state.sls('multiple-comps-repos')
    assert ret.failed is False
    for state in ret:
        assert state.result is True
        assert not state.changes

@pytest.fixture
def sources_list_file():
    if False:
        while True:
            i = 10
    fn_ = salt.utils.files.mkstemp(dir='/etc/apt/sources.list.d', suffix='.list')
    try:
        yield fn_
    finally:
        try:
            os.remove(fn_)
        except OSError:
            pass

def test_pkgrepo_with_architectures(pkgrepo, grains, sources_list_file, subtests):
    if False:
        i = 10
        return i + 15
    '\n    Test managing a repo with architectures specified\n    '
    name = 'deb {{arch}}http://foo.com/bar/latest {oscodename} main'.format(oscodename=grains['oscodename'])

    def _get_arch(arch):
        if False:
            i = 10
            return i + 15
        return f'[arch={arch}] ' if arch else ''

    def _run(arch=None, test=False):
        if False:
            return 10
        return pkgrepo.managed(name=name.format(arch=_get_arch(arch)), file=sources_list_file, refresh=False, test=test)
    with subtests.test('test=True'):
        ret = _run(test=True)
        assert ret.changes == {'repo': name.format(arch='')}
        assert 'would be' in ret.comment
        assert ret.result is None
    with subtests.test('test=False'):
        ret = _run()
        assert ret.changes == {'repo': name.format(arch='')}
        assert ret.comment.startswith('Configured')
        assert ret.result is True
    with subtests.test('test=True repeat'):
        ret = _run(test=True)
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True
    with subtests.test('test=False repeat'):
        ret = _run()
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True
    expected_changes = {'line': {'new': name.format(arch=_get_arch('amd64')), 'old': name.format(arch='')}, 'architectures': {'new': ['amd64'], 'old': []}}
    with subtests.test('test=True arch=amd64'):
        ret = _run(arch='amd64', test=True)
        assert ret.changes == expected_changes
        assert 'would be' in ret.comment
        assert ret.result is None
    with subtests.test('test=False arch=amd64'):
        ret = _run(arch='amd64')
        assert ret.changes == expected_changes
        assert ret.comment.startswith('Configured')
        assert ret.result is True
    with subtests.test('test=True arch=amd64 repeat'):
        ret = _run(arch='amd64', test=True)
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True
    with subtests.test('test=False arch=amd64 repeat'):
        ret = _run(arch='amd64')
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True
    expected_changes = {'line': {'new': name.format(arch=''), 'old': name.format(arch=_get_arch('amd64'))}, 'architectures': {'new': [], 'old': ['amd64']}}
    with subtests.test('test=True arch=None'):
        ret = _run(test=True)
        assert ret.changes == expected_changes
        assert 'would be' in ret.comment
        assert ret.result is None
    with subtests.test('test=False arch=None'):
        ret = _run()
        assert ret.changes == expected_changes
        assert ret.comment.startswith('Configured')
        assert ret.result is True
    with subtests.test('test=True arch=None repeat'):
        ret = _run(test=True)
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True
    with subtests.test('test=False arch=None repeat'):
        ret = _run()
        assert not ret.changes
        assert 'already' in ret.comment
        assert ret.result is True

@pytest.fixture(scope='module')
def repo_uri():
    if False:
        return 10
    yield 'http://www.deb-multimedia.org'

@pytest.fixture
def trailing_slash_repo_file(grains):
    if False:
        while True:
            i = 10
    if grains['os_family'] != 'Debian':
        pytest.skip("Test only applicable to Debian flavors, not '{}'".format(grains['osfinger']))
    repo_file_path = '/etc/apt/sources.list.d/trailing-slash.list'
    try:
        yield repo_file_path
    finally:
        try:
            os.unlink(repo_file_path)
        except OSError:
            pass

@pytest.mark.requires_salt_states('pkgrepo.managed', 'pkgrepo.absent')
def test_repo_present_absent_trailing_slash_uri(pkgrepo, repo_uri, trailing_slash_repo_file):
    if False:
        print('Hello World!')
    '\n    test adding and then removing a repo with a trailing slash in the uri\n    '
    repo_content = f'deb {repo_uri}/ stable main'
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False, clean_file=True)
    with salt.utils.files.fopen(trailing_slash_repo_file, 'r') as fp:
        file_content = fp.read()
    assert file_content.strip() == f'deb {repo_uri}/ stable main'
    assert ret.changes
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False)
    assert not ret.changes
    ret = pkgrepo.absent(name=repo_content)
    assert ret.result

@pytest.mark.requires_salt_states('pkgrepo.managed', 'pkgrepo.absent')
def test_repo_present_absent_no_trailing_slash_uri(pkgrepo, repo_uri, trailing_slash_repo_file):
    if False:
        print('Hello World!')
    '\n    test adding a repo with a trailing slash in the uri\n    '
    repo_content = f'deb {repo_uri} stable main'
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False, clean_file=True)
    with salt.utils.files.fopen(trailing_slash_repo_file, 'r') as fp:
        file_content = fp.read()
    assert file_content.strip() == repo_content
    assert ret.changes
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False)
    assert not ret.changes
    ret = pkgrepo.absent(name=repo_content)
    assert ret.result

@pytest.mark.requires_salt_states('pkgrepo.managed', 'pkgrepo.absent')
def test_repo_present_absent_no_trailing_slash_uri_add_slash(pkgrepo, repo_uri, trailing_slash_repo_file):
    if False:
        i = 10
        return i + 15
    '\n    test adding a repo without a trailing slash, and then running it\n    again with a trailing slash.\n    '
    repo_content = f'deb {repo_uri} stable main'
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False, clean_file=True)
    with salt.utils.files.fopen(trailing_slash_repo_file, 'r') as fp:
        file_content = fp.read()
    assert file_content.strip() == repo_content
    assert ret.changes
    repo_content = f'deb {repo_uri}/ stable main'
    ret = pkgrepo.managed(name=repo_content, file=trailing_slash_repo_file, refresh=False)
    with salt.utils.files.fopen(trailing_slash_repo_file, 'r') as fp:
        file_content = fp.read()
    assert file_content.strip() == repo_content
    ret = pkgrepo.absent(name=repo_content)
    assert ret.result

@pytest.mark.requires_salt_states('pkgrepo.absent')
def test_repo_absent_existing_repo_trailing_slash_uri(pkgrepo, repo_uri, subtests, trailing_slash_repo_file):
    if False:
        while True:
            i = 10
    '\n    Test pkgrepo.absent with a URI containing a trailing slash\n\n    This test is different from test_repo_present_absent_trailing_slash_uri.\n    That test first does a pkgrepo.managed with a URI containing a trailing\n    slash. Since pkgrepo.managed normalizes the URI by removing the trailing\n    slash, the resulting repo file created by Salt does not contain one. This\n    tests the case where Salt is asked to remove an existing repo with a\n    trailing slash in the repo URI.\n\n    See https://github.com/saltstack/salt/issues/64286\n    '
    repo_file = pathlib.Path(trailing_slash_repo_file)
    repo_content = f'deb [arch=amd64] {repo_uri}/ stable main'
    with subtests.test('Remove repo with trailing slash in URI'):
        repo_file.write_text(f'{repo_content}\n')
        ret = pkgrepo.absent(name=repo_content)
        assert ret.result
        assert ret.changes
        assert not repo_file.exists()
        ret = pkgrepo.absent(name=repo_content)
        assert ret.result
        assert not ret.changes
        assert not repo_file.exists()
    with subtests.test('URI match with mismatched arch'):
        repo_file.write_text(f'deb {repo_uri} stable main\n')
        ret = pkgrepo.absent(name=repo_content)
        assert ret.result
        assert not ret.changes
        assert repo_file.exists()

@attr.s(kw_only=True)
class Repo:
    key_root = attr.ib(default=pathlib.Path('/usr', 'share', 'keyrings'))
    signedby = attr.ib(default=False)
    grains = attr.ib()
    fullname = attr.ib()
    alt_repo = attr.ib(init=False)
    key_file = attr.ib()
    sources_list_file = attr.ib()
    repo_file = attr.ib()
    repo_content = attr.ib()
    key_url = attr.ib()

    @fullname.default
    def _default_fullname(self):
        if False:
            return 10
        return self.grains['osfullname'].lower().split()[0]

    @alt_repo.default
    def _default_alt_repo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use an alternative repo, packages do not\n        exist for the OS on repo.saltproject.io\n        '
        if self.grains['osfullname'] == 'Ubuntu' and self.grains['osrelease'] == '22.04' or ('Debian' in self.grains['osfullname'] and self.grains['osrelease'] == '12'):
            return True
        return False

    @key_file.default
    def _default_key_file(self):
        if False:
            while True:
                i = 10
        key_file = self.key_root / 'salt-archive-keyring.gpg'
        if self.alt_repo:
            key_file = self.key_root / 'elasticsearch-keyring.gpg'
        key_file.parent.mkdir(exist_ok=True)
        assert not key_file.is_file()
        return key_file

    @repo_file.default
    def _default_repo_file(self):
        if False:
            while True:
                i = 10
        return self.sources_list_file

    @repo_content.default
    def _default_repo_content(self):
        if False:
            return 10
        if self.alt_repo:
            opts = ' '
            if self.signedby:
                opts = ' [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] '
            repo_content = 'deb{}https://artifacts.elastic.co/packages/8.x/apt stable main'.format(opts)
        else:
            opts = '[arch={arch}]'.format(arch=self.grains['osarch'])
            if self.signedby:
                opts = '[arch={arch} signed-by=/usr/share/keyrings/salt-archive-keyring.gpg]'.format(arch=self.grains['osarch'])
            repo_content = 'deb {opts} https://repo.saltproject.io/py3/{}/{}/{arch}/latest {} main'.format(self.fullname, self.grains['osrelease'], self.grains['oscodename'], arch=self.grains['osarch'], opts=opts)
        return repo_content

    @key_url.default
    def _default_key_url(self):
        if False:
            for i in range(10):
                print('nop')
        key_url = 'https://repo.saltproject.io/py3/{}/{}/{}/latest/salt-archive-keyring.gpg'.format(self.fullname, self.grains['osrelease'], self.grains['osarch'])
        if self.alt_repo:
            key_url = 'https://artifacts.elastic.co/GPG-KEY-elasticsearch'
        return key_url

@pytest.fixture
def repo(request, grains, sources_list_file):
    if False:
        i = 10
        return i + 15
    signedby = False
    if 'signedby' in request.node.name:
        signedby = True
    repo = Repo(grains=grains, sources_list_file=sources_list_file, signedby=signedby)
    yield repo
    for key in [repo.key_file, repo.key_file.parent / 'salt-alt-key.gpg']:
        if key.is_file():
            key.unlink()

def test_adding_repo_file_signedby(pkgrepo, states, repo, subtests):
    if False:
        i = 10
        return i + 15
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby\n    '

    def _run(test=False):
        if False:
            i = 10
            return i + 15
        return states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, signedby=str(repo.key_file), key_url=repo.key_url, aptkey=False, test=test)
    ret = _run()
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == repo.repo_content
    assert repo.key_file.is_file()
    assert repo.repo_content in ret.comment
    with subtests.test('test=True'):
        ret = _run(test=True)
        assert ret.changes == {}

def test_adding_repo_file_signedby_invalid_name(pkgrepo, states, repo):
    if False:
        print('Hello World!')
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby and the name is invalid.\n    Ensure we raise an error.\n    '
    default_sources = pathlib.Path('/etc', 'apt', 'sources.list')
    with salt.utils.files.fopen(default_sources, 'r') as fp:
        pre_file_content = fp.read()
    ret = states.pkgrepo.managed(name=repo.repo_content.strip('deb'), file=str(repo.repo_file), clean_file=True, signedby=str(repo.key_file), key_url=repo.key_url, aptkey=False, test=False)
    assert 'Failed to configure repo' in ret.comment
    assert 'This must be the complete repo entry' in ret.comment
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert not file_content
    with salt.utils.files.fopen(default_sources, 'r') as fp:
        post_file_content = fp.read()
    assert pre_file_content == post_file_content

def test_adding_repo_file_signedby_keyserver(pkgrepo, states, repo):
    if False:
        i = 10
        return i + 15
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby with a keyserver\n    '
    ret = states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, signedby=str(repo.key_file), keyserver='keyserver.ubuntu.com', keyid='0E08A149DE57BFBE', aptkey=False)
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == repo.repo_content
    assert repo.key_file.is_file()

def test_adding_repo_file_keyserver_key_url(pkgrepo, states, repo):
    if False:
        i = 10
        return i + 15
    '\n    Test adding a repo file using pkgrepo.managed\n    and a key_url.\n    '
    ret = states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, keyserver='keyserver.ubuntu.com', key_url=repo.key_url)
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == repo.repo_content

def test_adding_repo_file_signedby_alt_file(pkgrepo, states, repo):
    if False:
        i = 10
        return i + 15
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby and then running again with\n    different key path.\n    '
    ret = states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, key_url=repo.key_url, aptkey=False)
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == repo.repo_content
    assert repo.key_file.is_file()
    assert repo.repo_content in ret.comment
    key_file = repo.key_file.parent / 'salt-alt-key.gpg'
    repo_content = 'deb [arch=amd64 signed-by={}] https://repo.saltproject.io/py3/debian/10/amd64/latest buster main'.format(str(key_file))
    ret = states.pkgrepo.managed(name=repo_content, file=str(repo.repo_file), clean_file=True, key_url=repo.key_url, aptkey=False)
    with salt.utils.files.fopen(str(repo.repo_file), 'r') as fp:
        file_content = fp.read()
        assert file_content.strip() == repo_content
        assert file_content.endswith('\n')
    assert key_file.is_file()
    assert repo_content in ret.comment

def test_adding_repo_file_signedby_fail_key_keyid(pkgrepo, states, repo, subtests, modules):
    if False:
        print('Hello World!')
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby and keyid when adding the key fails\n    an error is returned\n    '

    def _run(test=False):
        if False:
            print('Hello World!')
        return states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, signedby=str(repo.key_file), keyid='10857FFDD3F91EAE577A21D664CBBC8173D76B3F1', keyserver='keyserver.ubuntu.com', aptkey=False, test=test, keydir='/tmp/test')
    ret = _run()
    assert 'Failed to configure repo' in ret.comment
    assert 'Could not add key' in ret.comment

def test_adding_repo_file_signedby_fail_key_keyurl(pkgrepo, states, repo, subtests, modules):
    if False:
        while True:
            i = 10
    '\n    Test adding a repo file using pkgrepo.managed\n    and setting signedby and keyurl when adding the key fails\n    an error is returned\n    '

    def _run(test=False):
        if False:
            while True:
                i = 10
        with patch('salt.utils.path.which', MagicMock(side_effect=[True, True, False, False])):
            return states.pkgrepo.managed(name=repo.repo_content, file=str(repo.repo_file), clean_file=True, key_url='https://repo.saltproject.io/salt/py3/ubuntu/20.04/amd64/latest/SALT-PROJECT-GPG-PUBKEY-2023.pub', aptkey=False)
    ret = _run()
    assert 'Failed to configure repo' in ret.comment
    assert 'Could not add key' in ret.comment