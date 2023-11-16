import pytest
from reflex.components.media.icon import ICON_LIST, Icon
from reflex.utils import format

def test_no_tag_errors():
    if False:
        i = 10
        return i + 15
    'Test that an icon without a tag raises an error.'
    with pytest.raises(AttributeError):
        Icon.create()

def test_children_errors():
    if False:
        print('Hello World!')
    'Test that an icon with children raises an error.'
    with pytest.raises(AttributeError):
        Icon.create('child', tag='search')

@pytest.mark.parametrize('tag', ICON_LIST)
def test_valid_icon(tag: str):
    if False:
        i = 10
        return i + 15
    'Test that a valid icon does not raise an error.\n\n    Args:\n        tag: The icon tag.\n    '
    icon = Icon.create(tag=tag)
    assert icon.tag == format.to_title_case(tag) + 'Icon'

@pytest.mark.parametrize('tag', ['', ' ', 'invalid', 123])
def test_invalid_icon(tag):
    if False:
        print('Hello World!')
    'Test that an invalid icon raises an error.\n\n    Args:\n        tag: The icon tag.\n    '
    with pytest.raises(ValueError):
        Icon.create(tag=tag)

@pytest.mark.parametrize('tag', ['Check', 'Close', 'eDit'])
def test_tag_with_capital(tag: str):
    if False:
        while True:
            i = 10
    'Test that an icon that tag with capital does not raise an error.\n\n    Args:\n        tag: The icon tag.\n    '
    icon = Icon.create(tag=tag)
    assert icon.tag == format.to_title_case(tag) + 'Icon'