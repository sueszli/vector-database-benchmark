import logging
from errbot import rendering
log = logging.getLogger(__name__)

def test_ansi():
    if False:
        return 10
    mdc = rendering.ansi()
    assert mdc.convert('*woot*') == '\x1b[4mwoot\x1b[24m\x1b[0m'

def test_text():
    if False:
        while True:
            i = 10
    mdc = rendering.text()
    assert mdc.convert('*woot*') == 'woot'
    assert mdc.convert('# woot') == 'WOOT'

def test_mde2md():
    if False:
        i = 10
        return i + 15
    mdc = rendering.md()
    assert mdc.convert('woot') == 'woot'
    assert mdc.convert('woot{:stuff} really{:otherstuff}') == 'woot really'

def test_escaping():
    if False:
        return 10
    mdc = rendering.text()
    original = '#not a title\n*not italic*\n`not code`\ntoto{not annotation}'
    escaped = rendering.md_escape(original)
    assert original == mdc.convert(escaped)