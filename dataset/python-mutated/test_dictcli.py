import pathlib
import pytest
from qutebrowser.browser.webengine import spell
from qutebrowser.config import configdata
from scripts import dictcli

def afrikaans():
    if False:
        for i in range(10):
            print('nop')
    return dictcli.Language(code='af-ZA', name='Afrikaans (South Africa)', remote_filename='af-ZA-3-0.bdic')

def english():
    if False:
        return 10
    return dictcli.Language(code='en-US', name='English (United States)', remote_filename='en-US-7-1.bdic')

def polish():
    if False:
        i = 10
        return i + 15
    return dictcli.Language(code='pl-PL', name='Polish (Poland)', remote_filename='pl-PL-3-0.bdic')

def langs():
    if False:
        for i in range(10):
            print('nop')
    return [afrikaans(), english(), polish()]

@pytest.fixture(autouse=True)
def configdata_init():
    if False:
        while True:
            i = 10
    if configdata.DATA is None:
        configdata.init()

@pytest.fixture(autouse=True)
def dict_tmp_path(tmp_path, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(spell, 'dictionary_dir', lambda : str(tmp_path))
    return tmp_path

def test_language(dict_tmp_path):
    if False:
        while True:
            i = 10
    (dict_tmp_path / 'pl-PL-2-0.bdic').touch()
    assert english().local_filename is None
    assert polish()

def test_parse_entry():
    if False:
        while True:
            i = 10
    assert dictcli.parse_entry({'name': 'en-US-7-1.bdic'}) == ('en-US', 'en-US-7-1.bdic')

def test_latest_yet():
    if False:
        return 10
    code2file = {'en-US': 'en-US-7-1.bdic'}
    assert not dictcli.latest_yet(code2file, 'en-US', 'en-US-7-0.bdic')
    assert not dictcli.latest_yet(code2file, 'en-US', 'en-US-7-1.bdic')
    assert dictcli.latest_yet(code2file, 'en-US', 'en-US-8-0.bdic')

def test_available_languages(dict_tmp_path, monkeypatch):
    if False:
        i = 10
        return i + 15
    for f in ['pl-PL-2-0.bdic', english().remote_filename]:
        (dict_tmp_path / f).touch()
    monkeypatch.setattr(dictcli, 'language_list_from_api', lambda : [(lang.code, lang.remote_filename) for lang in langs()])
    languages = sorted(dictcli.available_languages(), key=lambda lang: lang.code)
    assert languages == [dictcli.Language(code='af-ZA', name='Afrikaans (South Africa)', remote_filename='af-ZA-3-0.bdic', local_filename=None), dictcli.Language(code='en-US', name='English (United States)', remote_filename='en-US-7-1.bdic', local_filename=None), dictcli.Language(code='pl-PL', name='Polish (Poland)', remote_filename='pl-PL-3-0.bdic', local_filename='pl-PL-2-0.bdic')]

def test_filter_languages():
    if False:
        return 10
    filtered_langs = dictcli.filter_languages(langs(), ['af-ZA'])
    assert filtered_langs == [afrikaans()]
    filtered_langs = dictcli.filter_languages(langs(), ['pl-PL', 'en-US'])
    assert filtered_langs == [english(), polish()]
    with pytest.raises(dictcli.InvalidLanguageError):
        dictcli.filter_languages(langs(), ['pl-PL', 'en-GB'])

def test_install(dict_tmp_path, monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(dictcli, 'download_dictionary', lambda _url, dest: pathlib.Path(dest).touch())
    dictcli.install(langs())
    installed_files = [f.name for f in dict_tmp_path.glob('*')]
    expected_files = [lang.remote_filename for lang in langs()]
    assert sorted(installed_files) == sorted(expected_files)

def test_update(dict_tmp_path, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(dictcli, 'download_dictionary', lambda _url, dest: pathlib.Path(dest).touch())
    (dict_tmp_path / 'pl-PL-2-0.bdic').touch()
    assert polish().local_version < polish().remote_version
    dictcli.update(langs())
    assert polish().local_version == polish().remote_version

def test_remove_old(dict_tmp_path, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(dictcli, 'download_dictionary', lambda _url, dest: pathlib.Path(dest).touch())
    for f in ['pl-PL-2-0.bdic', polish().remote_filename, english().remote_filename]:
        (dict_tmp_path / f).touch()
    dictcli.remove_old(langs())
    installed_files = [f.name for f in dict_tmp_path.glob('*')]
    expected_files = [polish().remote_filename, english().remote_filename]
    assert sorted(installed_files) == sorted(expected_files)