import os
import subprocess
import tempfile
from pathlib import Path
import pytest
SECUREDROP_ROOT = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DEB_PATHS = list((SECUREDROP_ROOT / 'build/focal').glob('*.deb'))
SITE_PACKAGES = '/opt/venvs/securedrop-app-code/lib/python3.8/site-packages'

@pytest.fixture(scope='module')
def securedrop_app_code_contents() -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the content listing of the securedrop-app-code Debian package.\n    '
    try:
        path = [pkg for pkg in DEB_PATHS if pkg.name.startswith('securedrop-app-code')][0]
    except IndexError:
        raise RuntimeError('Unable to find securedrop-app-code package in build/ folder')
    return subprocess.check_output(['dpkg-deb', '--contents', path]).decode()

@pytest.mark.parametrize('deb', DEB_PATHS)
def test_deb_packages_appear_installable(deb: Path) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Confirms that a dry-run of installation reports no errors.\n    Simple check for valid Debian package structure, but not thorough.\n    When run on a malformed package, `dpkg` will report:\n\n       dpkg-deb: error: `foo.deb' is not a debian format archive\n\n    Testing application behavior is left to the functional tests.\n    "
    path = os.getenv('PATH') + ':/usr/sbin:/sbin'
    subprocess.check_call(['dpkg', '--install', '--dry-run', deb], env={'PATH': path})

@pytest.mark.parametrize('deb', DEB_PATHS)
def test_deb_package_contains_expected_conffiles(deb: Path):
    if False:
        while True:
            i = 10
    '\n    Ensures the `securedrop-app-code` package declares only allow-listed\n    `conffiles`. Several files in `/etc/` would automatically be marked\n    conffiles, which would break unattended updates to critical package\n    functionality such as AppArmor profiles. This test validates overrides\n    in the build logic to unset those conffiles.\n\n    The same applies to `securedrop-config` too.\n    '
    if not deb.name.startswith(('securedrop-app-code', 'securedrop-config')):
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call(['dpkg-deb', '--control', deb, tmpdir])
        conffiles_path = Path(tmpdir) / 'conffiles'
        assert conffiles_path.exists()
        assert conffiles_path.read_text().rstrip() == ''

@pytest.mark.parametrize('path', ['/var/www/securedrop/.well-known/pki-validation/', '/var/www/securedrop/translations/messages.pot', '/var/www/securedrop/translations/de_DE/LC_MESSAGES/messages.mo', f'{SITE_PACKAGES}/redwood/redwood.cpython-38-x86_64-linux-gnu.so'])
def test_app_code_paths(securedrop_app_code_contents: str, path: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensures the `securedrop-app-code` package contains the specified paths\n    '
    for line in securedrop_app_code_contents.splitlines():
        if line.endswith(path):
            assert True
            return
    pytest.fail('not found')

@pytest.mark.parametrize('path', ['/var/www/securedrop/static/.webassets-cache/', '/var/www/securedrop/static/gen/', '/var/www/securedrop/config.py', '/var/www/securedrop/static/i/custom_logo.png', '.j2'])
def test_app_code_paths_missing(securedrop_app_code_contents: str, path: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensures the `securedrop-app-code` package do *NOT* contain the specified paths\n    '
    for line in securedrop_app_code_contents.splitlines():
        if line.endswith(path):
            pytest.fail(f'found {line}')