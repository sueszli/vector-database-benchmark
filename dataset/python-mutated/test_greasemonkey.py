"""Tests for qutebrowser.browser.greasemonkey."""
import textwrap
import logging
import pathlib
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.utils import usertypes, version
from qutebrowser.browser import greasemonkey
from qutebrowser.misc import objects
test_gm_script = '\n// ==UserScript==\n// @name qutebrowser test userscript\n// @namespace invalid.org\n// @include http://localhost:*/data/title.html\n// @match http://*.trolol.com/*\n// @exclude https://badhost.xxx/*\n// @run-at document-start\n// ==/UserScript==\nconsole.log("Script is running.");\n'
pytestmark = [pytest.mark.usefixtures('data_tmpdir'), pytest.mark.usefixtures('config_tmpdir')]

@pytest.fixture
def gm_manager(monkeypatch) -> greasemonkey.GreasemonkeyManager:
    if False:
        while True:
            i = 10
    gmm = greasemonkey.GreasemonkeyManager()
    monkeypatch.setattr(greasemonkey, 'gm_manager', gmm)
    return gmm

@pytest.fixture
def wrong_path_setup():
    if False:
        return 10
    wrong_path = _scripts_dir() / 'test1.user.js'
    wrong_path.mkdir()
    _save_script(test_gm_script, 'test2.user.js')

def _scripts_dir() -> pathlib.Path:
    if False:
        while True:
            i = 10
    p = pathlib.Path(greasemonkey._scripts_dirs()[0])
    p.mkdir(exist_ok=True)
    return p

def _save_script(script_text: str, filename: str) -> None:
    if False:
        while True:
            i = 10
    file_path = _scripts_dir() / filename
    file_path.write_text(script_text, encoding='utf-8')

def test_all(gm_manager):
    if False:
        print('Hello World!')
    'Test that a script gets read from file, parsed and returned.'
    name = 'qutebrowser test userscript'
    _save_script(test_gm_script, 'test.user.js')
    result = gm_manager.load_scripts()
    assert len(result.successful) == 1
    assert result.successful[0].name == name
    assert not result.errors
    all_scripts = gm_manager.all_scripts()
    assert len(all_scripts) == 1
    assert all_scripts[0].name == name

@pytest.mark.parametrize('header, expected', [([], {'name': 'test.user.js', 'namespace': None, 'includes': ['*'], 'matches': [], 'excludes': [], 'run_at': None, 'runs_on_sub_frames': True, 'jsworld': 'main'}), (['@include https://example.org'], {'includes': ['https://example.org'], 'excludes': [], 'matches': []}), (['@include https://example.org', '@include https://example.com'], {'includes': ['https://example.org', 'https://example.com'], 'excludes': [], 'matches': []}), (['@match https://example.org'], {'includes': [], 'excludes': [], 'matches': ['https://example.org']}), (['@match https://example.org', '@exclude_match https://example.com'], {'includes': [], 'excludes': ['https://example.com'], 'matches': ['https://example.org']}), (['@exclude https://example.org'], {'includes': ['*'], 'excludes': ['https://example.org'], 'matches': []}), (['@exclude https://example.org', '@exclude_match https://example.com'], {'includes': ['*'], 'excludes': ['https://example.org', 'https://example.com'], 'matches': []}), (['@name testfoo'], {'name': 'testfoo', 'namespace': None}), (['@namespace testbar'], {'name': 'test.user.js', 'namespace': 'testbar'}), (['@name testfoo', '@namespace testbar'], {'name': 'testfoo', 'namespace': 'testbar'}), (['@description Replace ads by cat pictures'], {'description': 'Replace ads by cat pictures'}), (['@noframes'], {'runs_on_sub_frames': False}), (['@noframes blabla'], {'runs_on_sub_frames': False}), (['@require stuff.js'], {'requires': ['stuff.js']}), (['@qute-js-world main'], {'jsworld': 'main'})])
def test_attributes(header, expected):
    if False:
        return 10
    lines = ['// ==UserScript==', *(f'// {line}' for line in header), '// ==/UserScript==', "console.log('Hello World')"]
    source = '\n'.join(lines)
    print(source)
    script = greasemonkey.GreasemonkeyScript.parse(source, filename='test.user.js')
    actual = {k: getattr(script, k) for k in expected}
    assert actual == expected

def test_load_error(gm_manager, wrong_path_setup):
    if False:
        i = 10
        return i + 15
    'Test behavior when a script fails loading.'
    name = 'qutebrowser test userscript'
    result = gm_manager.load_scripts()
    assert len(result.successful) == 1
    assert result.successful[0].name == name
    assert len(result.errors) == 1
    assert result.errors[0][0] == 'test1.user.js'
    all_scripts = gm_manager.all_scripts()
    assert len(all_scripts) == 1
    assert all_scripts[0].name == name

@pytest.mark.parametrize('url, expected_matches', [('http://trolol.com/', 1), ('http://aaaaaaaaaa.com/', 0), ('https://badhost.xxx/', 0)])
def test_get_scripts_by_url(gm_manager, url, expected_matches):
    if False:
        return 10
    'Check Greasemonkey include/exclude rules work.'
    _save_script(test_gm_script, 'test.user.js')
    gm_manager.load_scripts()
    scripts = gm_manager.scripts_for(QUrl(url))
    assert len(scripts.start + scripts.end + scripts.idle) == expected_matches

@pytest.mark.parametrize('url, expected_matches', [('https://github.com/qutebrowser/qutebrowser/', 1), ('http://aaaaaaaaaa.com/', 0), ('http://github.com/foo', 0)])
def test_regex_includes_scripts_for(gm_manager, url, expected_matches):
    if False:
        i = 10
        return i + 15
    'Ensure our GM @*clude support supports regular expressions.'
    gh_dark_example = textwrap.dedent('\n        // ==UserScript==\n        // @include     /^https?://((gist|guides|help|raw|status|developer)\\.)?github\\.com/((?!generated_pages\\/preview).)*$/\n        // @exclude     /https?://github\\.com/foo/\n        // @run-at document-start\n        // ==/UserScript==\n    ')
    _save_script(gh_dark_example, 'test.user.js')
    gm_manager.load_scripts()
    scripts = gm_manager.scripts_for(QUrl(url))
    assert len(scripts.start + scripts.end + scripts.idle) == expected_matches

def test_no_metadata(gm_manager, caplog):
    if False:
        for i in range(10):
            print('nop')
    'Run on all sites at document-end is the default.'
    _save_script('var nothing = true;\n', 'nothing.user.js')
    with caplog.at_level(logging.WARNING):
        gm_manager.load_scripts()
    scripts = gm_manager.scripts_for(QUrl('http://notamatch.invalid/'))
    assert len(scripts.start + scripts.end + scripts.idle) == 1
    assert len(scripts.end) == 1

def test_no_name():
    if False:
        while True:
            i = 10
    'Ensure that GreaseMonkeyScripts must have a name.'
    msg = '@name key required or pass filename to init.'
    with pytest.raises(ValueError, match=msg):
        greasemonkey.GreasemonkeyScript([('something', 'else')], '')

def test_no_name_with_fallback():
    if False:
        print('Hello World!')
    "Ensure that script's name can fallback to the provided filename."
    script = greasemonkey.GreasemonkeyScript([('something', 'else')], '', filename='C:\\COM1')
    assert script
    assert script.name == 'C:\\COM1'

@pytest.mark.parametrize('properties, inc_counter, expected', [([('name', 'gorilla')], False, 'GM-gorilla'), ([('namespace', 'apes'), ('name', 'gorilla')], False, 'GM-apes/gorilla'), ([('name', 'gorilla')], True, 'GM-gorilla-2'), ([('namespace', 'apes'), ('name', 'gorilla')], True, 'GM-apes/gorilla-2')])
def test_full_name(properties, inc_counter, expected):
    if False:
        print('Hello World!')
    script = greasemonkey.GreasemonkeyScript(properties, code='')
    if inc_counter:
        script.dedup_suffix += 1
    assert script.full_name() == expected

def test_bad_scheme(gm_manager, caplog):
    if False:
        i = 10
        return i + 15
    "qute:// isn't in the list of allowed schemes."
    _save_script('var nothing = true;\n', 'nothing.user.js')
    with caplog.at_level(logging.WARNING):
        gm_manager.load_scripts()
    scripts = gm_manager.scripts_for(QUrl('qute://settings'))
    assert len(scripts.start + scripts.end + scripts.idle) == 0

def test_load_emits_signal(gm_manager, qtbot):
    if False:
        i = 10
        return i + 15
    gm_manager.load_scripts()
    with qtbot.wait_signal(gm_manager.scripts_reloaded):
        gm_manager.load_scripts()

def test_utf8_bom(gm_manager):
    if False:
        i = 10
        return i + 15
    'Make sure UTF-8 BOMs are stripped from scripts.\n\n    If we don\'t strip them, we\'ll have a BOM in the middle of the file, causing\n    QtWebEngine to not catch the "// ==UserScript==" line.\n    '
    script = textwrap.dedent('\n        \ufeff// ==UserScript==\n        // @name qutebrowser test userscript\n        // ==/UserScript==\n    '.lstrip('\n'))
    _save_script(script, 'bom.user.js')
    gm_manager.load_scripts()
    scripts = gm_manager.all_scripts()
    assert len(scripts) == 1
    script = scripts[0]
    assert '// ==UserScript==' in script.code().splitlines()

class TestForceDocumentEnd:

    def _get_script(self, *, namespace, name):
        if False:
            i = 10
            return i + 15
        source = textwrap.dedent('\n            // ==UserScript==\n            // @namespace {}\n            // @name {}\n            // ==/UserScript==\n        '.format(namespace, name))
        _save_script(source, 'force.user.js')
        gm_manager = greasemonkey.GreasemonkeyManager()
        gm_manager.load_scripts()
        scripts = gm_manager.all_scripts()
        assert len(scripts) == 1
        return scripts[0]

    @pytest.mark.parametrize('namespace, name, force', [('http://userstyles.org', 'foobar', True), ('https://github.com/ParticleCore', 'Iridium', True), ('https://github.com/ParticleCore', 'Foo', False), ('https://example.org', 'Iridium', False)])
    def test_matching(self, monkeypatch, namespace, name, force):
        if False:
            for i in range(10):
                print('nop')
        'Test matching based on namespace/name.'
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebEngine)
        script = self._get_script(namespace=namespace, name=name)
        assert script.needs_document_end_workaround() == force

    @pytest.mark.parametrize('namespace, name', [('http://userstyles.org', 'foobar'), ('https://github.com/ParticleCore', 'Iridium'), ('https://github.com/ParticleCore', 'Foo'), ('https://example.org', 'Iridium')])
    def test_webkit(self, monkeypatch, namespace, name):
        if False:
            i = 10
            return i + 15
        monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebKit)
        script = self._get_script(namespace=namespace, name=name)
        assert not script.needs_document_end_workaround()

def test_required_scripts_are_included(gm_manager, download_stub, tmp_path):
    if False:
        i = 10
        return i + 15
    test_require_script = textwrap.dedent('\n        // ==UserScript==\n        // @name qutebrowser test userscript\n        // @namespace invalid.org\n        // @include http://localhost:*/data/title.html\n        // @match http://trolol*\n        // @exclude https://badhost.xxx/*\n        // @run-at document-start\n        // @require http://localhost/test.js\n        // ==/UserScript==\n        console.log("Script is running.");\n    ')
    _save_script(test_require_script, 'requiring.user.js')
    (tmp_path / 'test.js').write_text('REQUIRED SCRIPT', encoding='UTF-8')
    gm_manager.load_scripts()
    assert len(gm_manager._in_progress_dls) == 1
    for download in gm_manager._in_progress_dls:
        download.finished.emit()
    scripts = gm_manager.all_scripts()
    assert len(scripts) == 1
    assert 'REQUIRED SCRIPT' in scripts[0].code()
    assert 'Script is running.' in scripts[0].code()
    assert scripts[0].excludes

def test_window_isolation(js_tester, request):
    if False:
        return 10
    'Check that greasemonkey scripts get a shadowed global scope.'
    setup_script = "window.$ = 'global'"
    test_gm_script = greasemonkey.GreasemonkeyScript.parse(textwrap.dedent('\n            // ==UserScript==\n            // @name scopetest\n            // ==/UserScript==\n            // Check the thing the page set is set to the expected type\n            result.push(window.$);\n            result.push($);\n            // Now overwrite it\n            window.$ = \'shadowed\';\n            // And check everything is how the script would expect it to be\n            // after just writing to the "global" scope\n            result.push(window.$);\n            result.push($);\n        '))
    test_script = '\n'.join(['\n        const result = [];\n        ', test_gm_script.code(), '\n        // Now check that the actual global scope has\n        // not been overwritten\n        result.push(window.$);\n        result.push($);\n        // And return our findings\n        result;\n        '])
    expected = ['global', 'global', 'shadowed', 'shadowed', 'global', 'global']
    xfail = False
    if js_tester.tab.backend == usertypes.Backend.QtWebKit and version.qWebKitVersion() == '602.1':
        expected[-1] = 'shadowed'
        expected[-2] = 'shadowed'
        xfail = True
    js_tester.run(setup_script)
    js_tester.run(test_script, expected=expected)
    if xfail:
        pytest.xfail('Broken on WebKit 602.1')

def test_shared_window_proxy(js_tester):
    if False:
        while True:
            i = 10
    'Check that all scripts have access to the same window proxy.'
    test_script_a = greasemonkey.GreasemonkeyScript.parse(textwrap.dedent("\n            // ==UserScript==\n            // @name a\n            // ==/UserScript==\n            // Set a value from script a\n            window.$ = 'test';\n        ")).code()
    test_script_b = greasemonkey.GreasemonkeyScript.parse(textwrap.dedent('\n            // ==UserScript==\n            // @name b\n            // ==/UserScript==\n            // Check that the value is accessible from script b\n            return [window.$, $];\n        ')).code()
    js_tester.run(test_script_a)
    js_tester.run(test_script_b, expected=['test', 'test'])

@pytest.mark.parametrize('run_at, start, end, idle, with_warning', [('document-start', True, False, False, False), ('document-end', False, True, False, False), ('document-idle', False, False, True, False), ('', False, True, False, False), ('bla', False, True, False, True)])
def test_run_at(gm_manager, run_at, start, end, idle, with_warning, caplog):
    if False:
        while True:
            i = 10
    script = greasemonkey.GreasemonkeyScript.parse(textwrap.dedent(f'\n            // ==UserScript==\n            // @name run-at-tester\n            // @run-at {run_at}\n            // ==/UserScript==\n            return document.readyState;\n        '))
    if with_warning:
        with caplog.at_level(logging.WARNING):
            gm_manager.add_script(script)
        msg = 'Script run-at-tester has invalid run-at defined, defaulting to document-end'
        assert caplog.messages == [msg]
    else:
        gm_manager.add_script(script)
    assert gm_manager._run_start == ([script] if start else [])
    assert gm_manager._run_end == ([script] if end else [])
    assert gm_manager._run_idle == ([script] if idle else [])

@pytest.mark.parametrize('scripts, expected', [([], 'No Greasemonkey scripts loaded'), ([greasemonkey.GreasemonkeyScript(properties={}, code='', filename='test')], 'Loaded Greasemonkey scripts:\n\ntest'), ([greasemonkey.GreasemonkeyScript(properties={}, code='', filename='test1'), greasemonkey.GreasemonkeyScript(properties={}, code='', filename='test2')], 'Loaded Greasemonkey scripts:\n\ntest1\ntest2')])
def test_load_results_successful(scripts, expected):
    if False:
        print('Hello World!')
    results = greasemonkey.LoadResults()
    results.successful = scripts
    assert results.successful_str() == expected

@pytest.mark.parametrize('errors, expected', [([], None), ([('test', 'could not frobnicate')], 'Greasemonkey scripts failed to load:\n\ntest: could not frobnicate'), ([('test1', 'could not frobnicate'), ('test2', 'frobnicator borked')], 'Greasemonkey scripts failed to load:\n\ntest1: could not frobnicate\ntest2: frobnicator borked')])
def test_load_results_errors(errors, expected):
    if False:
        print('Hello World!')
    results = greasemonkey.LoadResults()
    results.errors = errors
    assert results.error_str() == expected

@pytest.mark.parametrize('quiet', [False, True])
def test_greasemonkey_reload(gm_manager, quiet, message_mock):
    if False:
        while True:
            i = 10
    _save_script(test_gm_script, 'test.user.js')
    assert not gm_manager.all_scripts()
    greasemonkey.greasemonkey_reload(quiet=quiet)
    assert gm_manager.all_scripts()
    if quiet:
        assert not message_mock.messages
    else:
        msg = 'Loaded Greasemonkey scripts:\n\nqutebrowser test userscript'
        assert message_mock.getmsg().text == msg

@pytest.mark.parametrize('quiet', [False, True])
def test_greasemonkey_reload_errors(gm_manager, caplog, message_mock, wrong_path_setup, quiet):
    if False:
        return 10
    assert not gm_manager.all_scripts()
    with caplog.at_level(logging.ERROR):
        greasemonkey.greasemonkey_reload(quiet=quiet)
    assert len(gm_manager.all_scripts()) == 1
    text = 'Greasemonkey scripts failed to load:\n\ntest1.user.js'
    msg = message_mock.messages[-1]
    assert msg.level == usertypes.MessageLevel.error
    assert msg.text.startswith(text)

def test_init(monkeypatch, message_mock):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(greasemonkey, 'gm_manager', None)
    _save_script(test_gm_script, 'test.user.js')
    greasemonkey.init()
    assert not message_mock.messages
    assert len(greasemonkey.gm_manager.all_scripts()) == 1

def test_init_errors(monkeypatch, message_mock, wrong_path_setup, caplog):
    if False:
        print('Hello World!')
    monkeypatch.setattr(greasemonkey, 'gm_manager', None)
    with caplog.at_level(logging.ERROR):
        greasemonkey.init()
    assert len(greasemonkey.gm_manager.all_scripts()) == 1
    msg = message_mock.getmsg(usertypes.MessageLevel.error)
    text = 'Greasemonkey scripts failed to load:\n\ntest1.user.js'
    assert msg.text.startswith(text)