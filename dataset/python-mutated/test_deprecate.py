import pytest
from PIL import _deprecate

@pytest.mark.parametrize('version, expected', [(11, 'Old thing is deprecated and will be removed in Pillow 11 \\(2024-10-15\\)\\. Use new thing instead\\.'), (None, 'Old thing is deprecated and will be removed in a future version\\. Use new thing instead\\.')])
def test_version(version, expected):
    if False:
        i = 10
        return i + 15
    with pytest.warns(DeprecationWarning, match=expected):
        _deprecate.deprecate('Old thing', version, 'new thing')

def test_unknown_version():
    if False:
        for i in range(10):
            print('nop')
    expected = 'Unknown removal version: 12345. Update PIL\\._deprecate\\?'
    with pytest.raises(ValueError, match=expected):
        _deprecate.deprecate('Old thing', 12345, 'new thing')

@pytest.mark.parametrize('deprecated, plural, expected', [('Old thing', False, 'Old thing is deprecated and should be removed\\.'), ('Old things', True, 'Old things are deprecated and should be removed\\.')])
def test_old_version(deprecated, plural, expected):
    if False:
        return 10
    expected = ''
    with pytest.raises(RuntimeError, match=expected):
        _deprecate.deprecate(deprecated, 1, plural=plural)

def test_plural():
    if False:
        for i in range(10):
            print('nop')
    expected = 'Old things are deprecated and will be removed in Pillow 11 \\(2024-10-15\\)\\. Use new thing instead\\.'
    with pytest.warns(DeprecationWarning, match=expected):
        _deprecate.deprecate('Old things', 11, 'new thing', plural=True)

def test_replacement_and_action():
    if False:
        print('Hello World!')
    expected = "Use only one of 'replacement' and 'action'"
    with pytest.raises(ValueError, match=expected):
        _deprecate.deprecate('Old thing', 11, replacement='new thing', action='Upgrade to new thing')

@pytest.mark.parametrize('action', ['Upgrade to new thing', 'Upgrade to new thing.'])
def test_action(action):
    if False:
        while True:
            i = 10
    expected = 'Old thing is deprecated and will be removed in Pillow 11 \\(2024-10-15\\)\\. Upgrade to new thing\\.'
    with pytest.warns(DeprecationWarning, match=expected):
        _deprecate.deprecate('Old thing', 11, action=action)

def test_no_replacement_or_action():
    if False:
        while True:
            i = 10
    expected = 'Old thing is deprecated and will be removed in Pillow 11 \\(2024-10-15\\)'
    with pytest.warns(DeprecationWarning, match=expected):
        _deprecate.deprecate('Old thing', 11)