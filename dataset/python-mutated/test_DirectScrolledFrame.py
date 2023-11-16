from direct.gui.DirectScrolledFrame import DirectScrolledFrame
import pytest

def test_set_scrollbar_width():
    if False:
        while True:
            i = 10
    w = 1
    frm = DirectScrolledFrame(scrollBarWidth=w)
    assert frm['scrollBarWidth'] == 1
    assert frm.verticalScroll['frameSize'] == (-w / 2.0, w / 2.0, -1, 1)
    assert frm.horizontalScroll['frameSize'] == (-1, 1, -w / 2.0, w / 2.0)
    frm.verticalScroll['frameSize'] = (-2, 2, -4, 4)
    frm.horizontalScroll['frameSize'] = (-4, 4, -2, 2)
    assert frm.verticalScroll['frameSize'] == (-2, 2, -4, 4)
    assert frm.horizontalScroll['frameSize'] == (-4, 4, -2, 2)
    w = 2
    frm['scrollBarWidth'] = w
    assert frm['scrollBarWidth'] == 2
    assert frm.verticalScroll['frameSize'] == (-w / 2.0, w / 2.0, -4, 4)
    assert frm.horizontalScroll['frameSize'] == (-4, 4, -w / 2.0, w / 2.0)

def test_set_scrollbar_width_on_init():
    if False:
        while True:
            i = 10
    frm = DirectScrolledFrame(verticalScroll_frameSize=(-2, 2, -4, 4), horizontalScroll_frameSize=(-4, 4, -2, 2))
    assert frm.verticalScroll['frameSize'] == (-2, 2, -4, 4)
    assert frm.horizontalScroll['frameSize'] == (-4, 4, -2, 2)