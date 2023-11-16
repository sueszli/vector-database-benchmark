"""Testing dirstack"""
import os
import os.path
import subprocess
import sys
import pytest
from xonsh import dirstack
from xonsh.dirstack import DIRSTACK, _unc_tempDrives
from xonsh.platform import ON_WINDOWS
HERE = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.dirname(HERE)

def drive_in_use(letter):
    if False:
        while True:
            i = 10
    return ON_WINDOWS and os.system(f'vol {letter}: 2>nul>nul') == 0
MAX_TEMP_DRIVES = 4
TEMP_DRIVE = []
for d in 'zyxwvuts':
    if not drive_in_use(d):
        TEMP_DRIVE.append(d + ':')
pytestmark = pytest.mark.skipif(len(TEMP_DRIVE) < MAX_TEMP_DRIVES, reason='Too many drive letters are already used by Windows to run the tests.')
xfail_py310 = pytest.mark.xfail(sys.version_info >= (3, 10), reason='throws file-not-found error (todo: fix)')

@pytest.fixture(scope='module')
def shares_setup(tmpdir_factory):
    if False:
        i = 10
        return i + 15
    'create some shares to play with on current machine.\n\n    Yield (to test case) array of structs: [uncPath, driveLetter, equivLocalPath]\n\n    Side effect: `os.chdir(TEST_WORK_DIR)`\n    '
    if not ON_WINDOWS:
        return []
    shares = [['uncpushd_test_HERE', TEMP_DRIVE[1], HERE], ['uncpushd_test_PARENT', TEMP_DRIVE[3], PARENT]]
    for (s, d, l) in shares:
        rtn = subprocess.call(['NET', 'SHARE', s, '/delete'], universal_newlines=True)
        if rtn != 0:
            yield None
            return
        rtn = subprocess.call(['NET', 'SHARE', s + '=' + l], universal_newlines=True)
        if rtn != 0:
            yield None
            return
        rtn = subprocess.call(['NET', 'USE', d, '\\\\localhost' + '\\' + s], universal_newlines=True)
        if rtn != 0:
            yield None
            return
    yield [['\\\\localhost' + '\\' + s[0], s[1], s[2]] for s in shares]
    os.chdir(HERE)
    for dl in _unc_tempDrives:
        subprocess.call(['net', 'use', dl, '/delete'], universal_newlines=True)
    for (_, d, _) in shares:
        subprocess.call(['net', 'use', d, '/delete'], universal_newlines=True)

def test_pushdpopd(xession):
    if False:
        print('Hello World!')
    "Simple non-UNC push/pop to verify we didn't break nonUNC case."
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    dirstack.cd([PARENT])
    owd = os.getcwd()
    assert owd.casefold() == xession.env['PWD'].casefold()
    dirstack.pushd([HERE])
    wd = os.getcwd()
    assert wd.casefold() == HERE.casefold()
    dirstack.popd([])
    assert owd.casefold() == os.getcwd().casefold(), 'popd returned cwd to expected dir'

def test_cd_dot(xession):
    if False:
        return 10
    xession.env.update(dict(PWD=os.getcwd()))
    owd = os.getcwd().casefold()
    dirstack.cd(['.'])
    assert owd == os.getcwd().casefold()

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_simple_push_pop(xession, shares_setup):
    if False:
        return 10
    if shares_setup is None:
        return
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    dirstack.cd([PARENT])
    owd = os.getcwd()
    assert owd.casefold() == xession.env['PWD'].casefold()
    dirstack.pushd(['\\\\localhost\\uncpushd_test_HERE'])
    wd = os.getcwd()
    assert os.path.splitdrive(wd)[0].casefold() == TEMP_DRIVE[0]
    assert os.path.splitdrive(wd)[1].casefold() == '\\'
    dirstack.popd([])
    assert owd.casefold() == os.getcwd().casefold(), 'popd returned cwd to expected dir'
    assert len(_unc_tempDrives) == 0

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_push_to_same_share(xession, shares_setup):
    if False:
        while True:
            i = 10
    if shares_setup is None:
        return
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    dirstack.cd([PARENT])
    owd = os.getcwd()
    assert owd.casefold() == xession.env['PWD'].casefold()
    dirstack.pushd(['\\\\localhost\\uncpushd_test_HERE'])
    wd = os.getcwd()
    assert os.path.splitdrive(wd)[0].casefold() == TEMP_DRIVE[0]
    assert os.path.splitdrive(wd)[1].casefold() == '\\'
    assert len(_unc_tempDrives) == 1
    assert len(DIRSTACK) == 1
    dirstack.pushd(['\\\\localhost\\uncpushd_test_HERE'])
    wd = os.getcwd()
    assert os.path.splitdrive(wd)[0].casefold() == TEMP_DRIVE[0]
    assert os.path.splitdrive(wd)[1].casefold() == '\\'
    assert len(_unc_tempDrives) == 1
    assert len(DIRSTACK) == 2
    dirstack.popd([])
    assert os.path.isdir(TEMP_DRIVE[0] + '\\'), 'Temp drive not unmapped till last reference removed'
    dirstack.popd([])
    assert owd.casefold() == os.getcwd().casefold(), 'popd returned cwd to expected dir'
    assert len(_unc_tempDrives) == 0

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_push_other_push_same(xession, shares_setup):
    if False:
        while True:
            i = 10
    'push to a, then to b. verify drive letter is TEMP_DRIVE[2], skipping already used TEMP_DRIVE[1]\n    Then push to a again. Pop (check b unmapped and a still mapped), pop, pop (check a is unmapped)\n    '
    if shares_setup is None:
        return
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    dirstack.cd([PARENT])
    owd = os.getcwd()
    assert owd.casefold() == xession.env['PWD'].casefold()
    dirstack.pushd(['\\\\localhost\\uncpushd_test_HERE'])
    assert os.getcwd().casefold() == TEMP_DRIVE[0] + '\\'
    assert len(_unc_tempDrives) == 1
    assert len(DIRSTACK) == 1
    dirstack.pushd(['\\\\localhost\\uncpushd_test_PARENT'])
    os.getcwd()
    assert os.getcwd().casefold() == TEMP_DRIVE[2] + '\\'
    assert len(_unc_tempDrives) == 2
    assert len(DIRSTACK) == 2
    dirstack.pushd(['\\\\localhost\\uncpushd_test_HERE'])
    assert os.getcwd().casefold() == TEMP_DRIVE[0] + '\\'
    assert len(_unc_tempDrives) == 2
    assert len(DIRSTACK) == 3
    dirstack.popd([])
    assert os.getcwd().casefold() == TEMP_DRIVE[2] + '\\'
    assert len(_unc_tempDrives) == 2
    assert len(DIRSTACK) == 2
    assert os.path.isdir(TEMP_DRIVE[2] + '\\')
    assert os.path.isdir(TEMP_DRIVE[0] + '\\')
    dirstack.popd([])
    assert os.getcwd().casefold() == TEMP_DRIVE[0] + '\\'
    assert len(_unc_tempDrives) == 1
    assert len(DIRSTACK) == 1
    assert not os.path.isdir(TEMP_DRIVE[2] + '\\')
    assert os.path.isdir(TEMP_DRIVE[0] + '\\')
    dirstack.popd([])
    assert os.getcwd().casefold() == owd.casefold()
    assert len(_unc_tempDrives) == 0
    assert len(DIRSTACK) == 0
    assert not os.path.isdir(TEMP_DRIVE[2] + '\\')
    assert not os.path.isdir(TEMP_DRIVE[0] + '\\')

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_push_base_push_rempath(xession):
    if False:
        i = 10
        return i + 15
    'push to subdir under share, verify  mapped path includes subdir'
    pass

@pytest.fixture
def toggle_unc_check():
    if False:
        print('Hello World!')
    old_wval = None

    def _update_key(key_type, value: int):
        if False:
            i = 10
            return i + 15
        import winreg
        old_wval = 0
        with dirstack._win_reg_key(key_type, 'software\\microsoft\\command processor', access=winreg.KEY_WRITE) as key:
            try:
                (wval, wtype) = winreg.QueryValueEx(key, 'DisableUNCCheck')
                old_wval = wval
            except OSError:
                pass
            winreg.SetValueEx(key, 'DisableUNCCheck', None, winreg.REG_DWORD, value)
        return old_wval

    def update(value: int):
        if False:
            while True:
                i = 10
        import winreg
        for key_type in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                return _update_key(key_type, value)
            except OSError:
                pass

    def wrapper(value: int):
        if False:
            i = 10
            return i + 15
        nonlocal old_wval
        old_wval = update(value) or 0
        return old_wval
    yield wrapper
    if old_wval is not None:
        update(old_wval)

@pytest.fixture()
def with_unc_check_enabled(toggle_unc_check):
    if False:
        print('Hello World!')
    return toggle_unc_check(1)

@pytest.fixture()
def with_unc_check_disabled(toggle_unc_check):
    if False:
        print('Hello World!')
    return toggle_unc_check(0)

@pytest.fixture()
def xonsh_builtins_cd(xession):
    if False:
        for i in range(10):
            print('nop')
    xession.env['CDPATH'] = PARENT
    xession.env['PWD'] = os.getcwd()
    xession.env['DIRSTACK_SIZE'] = 20
    return xession

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_cd_unc_auto_pushd(xonsh_builtins_cd, with_unc_check_enabled):
    if False:
        i = 10
        return i + 15
    xonsh_builtins_cd.env['AUTO_PUSHD'] = True
    (so, se, rc) = dirstack.cd(['\\\\localhost\\uncpushd_test_PARENT'])
    if rc != 0:
        return
    assert os.getcwd().casefold() == TEMP_DRIVE[0] + '\\'
    assert len(DIRSTACK) == 1
    assert os.path.isdir(TEMP_DRIVE[0] + '\\')

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_cd_unc_nocheck(xonsh_builtins_cd, with_unc_check_disabled):
    if False:
        return 10
    if with_unc_check_disabled == 0:
        return
    dirstack.cd(['\\\\localhost\\uncpushd_test_HERE'])
    assert os.getcwd().casefold() == '\\\\localhost\\uncpushd_test_here'

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_cd_unc_no_auto_pushd(xonsh_builtins_cd, with_unc_check_enabled):
    if False:
        i = 10
        return i + 15
    if with_unc_check_enabled == 0:
        return
    (so, se, rc) = dirstack.cd(['\\\\localhost\\uncpushd_test_PARENT'])
    assert rc != 0
    assert so is None or len(so) == 0
    assert 'disableunccheck' in se.casefold() and 'auto_pushd' in se.casefold()

@pytest.mark.skipif(not ON_WINDOWS, reason='Windows-only UNC functionality')
def test_uncpushd_unc_check():
    if False:
        return 10
    pass