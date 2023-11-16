import os
import sys
from unittest import mock
import ctypes
import pytest
from apprise import AppriseLocale
from apprise.utils import environ
from importlib import reload
import logging
logging.disable(logging.CRITICAL)

def test_apprise_trans():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: Test apprise locale object\n    '
    lazytrans = AppriseLocale.LazyTranslation('Token')
    assert str(lazytrans) == 'Token'

@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
def test_apprise_trans_gettext_init():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: Handle gettext\n    '
    AppriseLocale.GETTEXT_LOADED = False
    al = AppriseLocale.AppriseLocale()
    with al.lang_at('en') as _:
        assert _ is None
    AppriseLocale.GETTEXT_LOADED = True

@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
@mock.patch('gettext.translation')
@mock.patch('locale.getlocale')
def test_apprise_trans_gettext_translations(mock_getlocale, mock_gettext_trans):
    if False:
        print('Hello World!')
    '\n    API: Apprise() Gettext translations\n\n    '
    mock_getlocale.return_value = ('en_US', 'UTF-8')
    mock_gettext_trans.side_effect = FileNotFoundError()
    al = AppriseLocale.AppriseLocale()
    with al.lang_at('en'):
        pass
    AppriseLocale.AppriseLocale(language='fr')

@pytest.mark.skipif(hasattr(ctypes, 'windll'), reason='Unique Nux test cases')
@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
@mock.patch('locale.getlocale')
def test_apprise_trans_gettext_lang_at(mock_getlocale):
    if False:
        i = 10
        return i + 15
    '\n    API: Apprise() Gettext lang_at\n\n    '
    mock_getlocale.return_value = ('en_CA', 'UTF-8')
    al = AppriseLocale.AppriseLocale()
    assert al.add('en', set_default=False) is True
    assert al.add('en', set_default=True) is True
    with al.lang_at('en'):
        pass
    AppriseLocale.AppriseLocale(language='fr')
    with al.lang_at('en') as _:
        assert callable(_)
    with al.lang_at('es') as _:
        assert callable(_)
    with al.lang_at('fr') as _:
        assert callable(_)
    fallback = AppriseLocale.AppriseLocale._default_language
    mock_getlocale.return_value = None
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
        AppriseLocale.AppriseLocale._default_language = 'zz'
        assert AppriseLocale.AppriseLocale.detect_language() is None
        al = AppriseLocale.AppriseLocale()
        assert al.lang is None
        with al.lang_at(None) as _:
            assert callable(_)
        with al.lang_at('en') as _:
            assert callable(_)
        with al.lang_at('es') as _:
            assert callable(_)
        with al.lang_at('fr') as _:
            assert callable(_)
        assert al.gettext('test') == 'test'
    with environ('LANGUAGE', 'LC_CTYPE', LC_ALL='C.UTF-8', LANG='en_CA'):
        AppriseLocale.AppriseLocale._default_language = 'fr'
        assert AppriseLocale.AppriseLocale.detect_language() == 'en'
        al = AppriseLocale.AppriseLocale()
        assert al.lang == 'en'
        assert al.gettext('test') == 'test'
        assert al.add('zy', set_default=False) is False
        assert al.gettext('test') == 'test'
        al.add('ab', set_default=True)
        assert al.gettext('test') == 'test'
        assert al.add('zy', set_default=False) is False
    AppriseLocale.AppriseLocale._default_language = fallback

@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
def test_apprise_trans_add():
    if False:
        return 10
    '\n    API: Apprise() Gettext add\n\n    '
    al = AppriseLocale.AppriseLocale()
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
        assert al.add('en') is True
    al = AppriseLocale.AppriseLocale()
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', LANG='C.UTF-8'):
        assert al.add('en') is True
    al = AppriseLocale.AppriseLocale()
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', LANG='en_CA.UTF-8'):
        assert al.add('en') is True
        assert al.add('en') is True
    assert al.add('bad') is False

@pytest.mark.skipif(not hasattr(ctypes, 'windll'), reason='Unique Windows test cases')
@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
@mock.patch('locale.getlocale')
def test_apprise_trans_windows_users_win(mock_getlocale):
    if False:
        while True:
            i = 10
    '\n    API: Apprise() Windows Locale Testing (Win version)\n\n    '
    mock_getlocale.return_value = ('fr_CA', 'UTF-8')
    with mock.patch('ctypes.windll.kernel32.GetUserDefaultUILanguage') as ui_lang:
        ui_lang.return_value = 4105
        with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
            AppriseLocale.AppriseLocale._default_language = 'zz'
            assert AppriseLocale.AppriseLocale.detect_language() == 'en'
        with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', LANG='es_AR'):
            assert AppriseLocale.AppriseLocale.detect_language() == 'es'
        with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
            assert AppriseLocale.AppriseLocale.detect_language() == 'en'
        assert AppriseLocale.AppriseLocale.detect_language(detect_fallback=False) is None
        ui_lang.return_value = 0
        with environ('LANGUAGE', 'LANG', 'LC_ALL', 'LC_CTYPE'):
            assert AppriseLocale.AppriseLocale.detect_language() == 'fr'

@pytest.mark.skipif(hasattr(ctypes, 'windll'), reason='Unique Nux test cases')
@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
@mock.patch('locale.getlocale')
def test_apprise_trans_windows_users_nux(mock_getlocale):
    if False:
        return 10
    '\n    API: Apprise() Windows Locale Testing (Nux version)\n\n    '
    mock_getlocale.return_value = ('fr_CA', 'UTF-8')
    windll = mock.Mock()
    setattr(ctypes, 'windll', windll)
    windll.kernel32.GetUserDefaultUILanguage.return_value = 4105
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
        AppriseLocale.AppriseLocale._default_language = 'zz'
        assert AppriseLocale.AppriseLocale.detect_language() == 'en'
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', LANG='es_AR'):
        assert AppriseLocale.AppriseLocale.detect_language() == 'es'
    with environ('LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG'):
        assert AppriseLocale.AppriseLocale.detect_language() == 'en'
    assert AppriseLocale.AppriseLocale.detect_language(detect_fallback=False) is None
    windll.kernel32.GetUserDefaultUILanguage.return_value = 0
    with environ('LANGUAGE', 'LANG', 'LC_ALL', 'LC_CTYPE'):
        assert AppriseLocale.AppriseLocale.detect_language() == 'fr'
    delattr(ctypes, 'windll')

@pytest.mark.skipif(sys.platform == 'win32', reason='Unique Nux test cases')
@mock.patch('locale.getlocale')
def test_detect_language_using_env(mock_getlocale):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the reading of information from an environment variable\n    '
    mock_getlocale.return_value = ('en_CA', 'UTF-8')
    with environ('LANG', 'LANGUAGE', 'LC_ALL', 'LC_CTYPE'):
        assert isinstance(AppriseLocale.AppriseLocale.detect_language(), str)
    with environ('LANGUAGE', 'LC_ALL', LC_CTYPE='garbage', LANG='fr_CA'):
        assert AppriseLocale.AppriseLocale.detect_language() == 'fr'
    with environ(*list(os.environ.keys()), LC_CTYPE='UTF-8'):
        assert isinstance(AppriseLocale.AppriseLocale.detect_language(), str)
    with environ(*list(os.environ.keys())):
        assert isinstance(AppriseLocale.AppriseLocale.detect_language(), str)
    mock_getlocale.return_value = None
    with environ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
        assert AppriseLocale.AppriseLocale.detect_language() is None
    mock_getlocale.return_value = (None, None)
    with environ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
        assert AppriseLocale.AppriseLocale.detect_language() is None
    AppriseLocale.AppriseLocale()

@pytest.mark.skipif('gettext' not in sys.modules, reason='Requires gettext')
def test_apprise_trans_gettext_missing(tmpdir):
    if False:
        while True:
            i = 10
    '\n    Verify we can still operate without the gettext library\n    '
    del sys.modules['gettext']
    gettext_dir = tmpdir.mkdir('gettext')
    gettext_dir.join('__init__.py').write('')
    gettext_dir.join('gettext.py').write('raise ImportError()')
    sys.path.insert(0, str(gettext_dir))
    reload(sys.modules['apprise.AppriseLocale'])
    from apprise import AppriseLocale
    assert AppriseLocale.GETTEXT_LOADED is False
    sys.path.pop(0)
    reload(sys.modules['apprise.AppriseLocale'])
    from apprise import AppriseLocale
    assert AppriseLocale.GETTEXT_LOADED is True