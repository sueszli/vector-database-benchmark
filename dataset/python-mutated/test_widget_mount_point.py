import pytest
from textual.widget import MountError, Widget

class Content(Widget):
    pass

class Body(Widget):
    pass

def test_find_dom_spot():
    if False:
        i = 10
        return i + 15
    screen = Widget(name='Screen')
    header = Widget(name='Header', id='header')
    body = Body(id='body')
    content = [Content(id=f'item{n}') for n in range(1000)]
    body._add_children(*content)
    footer = Widget(name='Footer', id='footer')
    screen._add_children(header, body, footer)
    assert list(screen._nodes) == [header, body, footer]
    assert screen._find_mount_point(1) == (screen, 1)
    assert screen._find_mount_point(body) == screen._find_mount_point(1)
    assert screen._find_mount_point('Body') == screen._find_mount_point(body)
    assert screen._find_mount_point('#body') == screen._find_mount_point(1)
    with pytest.raises(MountError):
        _ = screen._find_mount_point(Widget())