import io
from textwrap import dedent
import pytest
import salt.modules.solaris_shadow as solaris_shadow
from tests.support.mock import MagicMock, patch
try:
    import pwd
    missing_pwd = False
except ImportError:
    pwd = None
    missing_pwd = True
try:
    import spwd
    missing_spwd = False
except ImportError:
    missing_spwd = True
skip_on_missing_spwd = pytest.mark.skipif(missing_spwd, reason='Has no spwd module for accessing /etc/shadow passwords')
skip_on_missing_pwd = pytest.mark.skipif(missing_pwd, reason='Has no pwd module for accessing /etc/password passwords')

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {solaris_shadow: {'pwd': pwd}}

@pytest.fixture
def fake_fopen_has_etc_shadow():
    if False:
        print('Hello World!')
    contents = dedent('            foo:bar:bang\n            whatever:is:shadow\n            roscivs:bottia:bloop\n        ')
    fake_output_shadow_file = io.StringIO()

    def fopen(file, mode, *args, **kwargs):
        if False:
            print('Hello World!')
        for line in contents.split():
            if 'b' in mode:
                return io.BytesIO(contents.encode())
            elif 'w' in mode:
                return fake_output_shadow_file
            else:
                return io.StringIO(contents)
    with patch('salt.utils.files.fopen', side_effect=fopen, autospec=True):
        with patch.object(fake_output_shadow_file, 'close'):
            yield fake_output_shadow_file
            fake_output_shadow_file.close()

@pytest.fixture
def has_spwd():
    if False:
        print('Hello World!')
    with patch.object(solaris_shadow, 'HAS_SPWD', True):
        yield

@pytest.fixture
def has_not_spwd():
    if False:
        print('Hello World!')
    with patch.object(solaris_shadow, 'HAS_SPWD', False):
        yield

@pytest.fixture
def fake_spnam():
    if False:
        for i in range(10):
            print('nop')
    with patch('spwd.getspnam', autospec=True) as fake_spnam:
        yield fake_spnam

@pytest.fixture
def fake_pwnam():
    if False:
        for i in range(10):
            print('nop')
    with patch('pwd.getpwnam', autospec=True) as fake_pwnam:
        yield fake_pwnam

@pytest.fixture
def has_shadow_file():
    if False:
        i = 10
        return i + 15
    with patch('os.path.isfile', return_value=True):
        yield

@pytest.fixture
def has_not_shadow_file():
    if False:
        print('Hello World!')
    with patch('os.path.isfile', return_value=False):
        yield

@skip_on_missing_spwd
def test_when_spwd_module_exists_results_should_be_returned_from_getspnam(has_spwd, fake_spnam):
    if False:
        print('Hello World!')
    expected_results = {'name': 'roscivs', 'passwd': 'bottia', 'lstchg': '2010-08-14', 'min': 0, 'max': 42, 'warn': 'nope', 'inact': 'whatever', 'expire': 'never!'}
    fake_spnam.return_value.sp_nam = expected_results['name']
    fake_spnam.return_value.sp_pwd = expected_results['passwd']
    fake_spnam.return_value.sp_lstchg = expected_results['lstchg']
    fake_spnam.return_value.sp_min = expected_results['min']
    fake_spnam.return_value.sp_max = expected_results['max']
    fake_spnam.return_value.sp_warn = expected_results['warn']
    fake_spnam.return_value.sp_inact = expected_results['inact']
    fake_spnam.return_value.sp_expire = expected_results['expire']
    actual_results = solaris_shadow.info(name='roscivs')
    assert actual_results == expected_results

@skip_on_missing_spwd
def test_when_swpd_module_exists_and_no_results_then_results_should_be_empty(has_spwd, fake_spnam):
    if False:
        for i in range(10):
            print('nop')
    expected_results = {'name': '', 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
    fake_spnam.side_effect = KeyError
    actual_results = solaris_shadow.info(name='roscivs')
    assert actual_results == expected_results

@skip_on_missing_pwd
def test_when_pwd_fallback_is_used_and_no_name_exists_results_should_be_empty(has_not_spwd, fake_pwnam):
    if False:
        for i in range(10):
            print('nop')
    expected_results = {'name': '', 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
    fake_pwnam.side_effect = KeyError
    actual_results = solaris_shadow.info(name='wayne')
    assert actual_results == expected_results

@skip_on_missing_pwd
def test_when_etc_shadow_does_not_exist_info_should_be_empty_except_for_name(has_not_spwd, fake_pwnam, has_not_shadow_file):
    if False:
        return 10
    expected_results = {'name': 'wayne', 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
    fake_pwnam.return_value.pw_name = 'not this name'
    actual_results = solaris_shadow.info(name='wayne')
    assert actual_results == expected_results

@skip_on_missing_pwd
def test_when_etc_shadow_exists_but_name_not_in_shadow_passwd_field_should_be_empty(fake_fopen_has_etc_shadow, has_not_spwd, fake_pwnam, has_shadow_file):
    if False:
        print('Hello World!')
    with patch.dict(solaris_shadow.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 42})}):
        actual_result = solaris_shadow.info(name='badname')
    assert actual_result['passwd'] == ''

@skip_on_missing_pwd
def test_when_name_in_etc_shadow_passwd_should_be_in_info(fake_fopen_has_etc_shadow, has_not_spwd, fake_pwnam, has_shadow_file):
    if False:
        while True:
            i = 10
    with patch.dict(solaris_shadow.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 42})}):
        actual_result = solaris_shadow.info(name='roscivs')
    assert actual_result['passwd'] == 'bottia'

def test_when_set_password_and_not_has_shadow_ret_should_be_empty_dict(has_not_shadow_file):
    if False:
        i = 10
        return i + 15
    actual_result = solaris_shadow.set_password(name='fnord', password='blarp')
    assert actual_result == {}

def test_set_password_should_return_False_if_passwd_in_info_is_different_than_new_password(has_shadow_file, fake_fopen_has_etc_shadow):
    if False:
        return 10
    existing_password = 'Fnord'
    failed_set_password = 'ignore me'
    with patch('salt.modules.solaris_shadow.info', autospec=True, return_value={'passwd': existing_password}):
        actual_result = solaris_shadow.set_password(name='roscivs', password=failed_set_password)
        assert actual_result == False

@skip_on_missing_spwd
def test_when_set_password_and_name_in_shadow_then_password_should_be_changed_for_that_user(has_shadow_file, fake_fopen_has_etc_shadow, has_spwd, fake_spnam):
    if False:
        return 10
    expected_password = 'bottia2'
    expected_shadow_contents = dedent('            foo:bar:bang\n            whatever:is:shadow\n            roscivs:bottia2:bloop\n        ')
    with patch('salt.modules.solaris_shadow.info', autospec=True, return_value={'passwd': expected_password}):
        actual_result = solaris_shadow.set_password(name='roscivs', password=expected_password)
    assert fake_fopen_has_etc_shadow.getvalue() == expected_shadow_contents
    assert actual_result == True