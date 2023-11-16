from panda3d.core import ButtonHandle
from panda3d.core import GamepadButton
from panda3d.core import KeyboardButton
from panda3d.core import MouseButton

def test_buttonhandle_type():
    if False:
        for i in range(10):
            print('nop')
    assert ButtonHandle.get_class_type().name == 'ButtonHandle'

def test_buttonhandle_none():
    if False:
        while True:
            i = 10
    none = ButtonHandle.none()
    assert none.index == 0
    assert none.name == 'none'
    assert none == ButtonHandle.none()
    assert none.alias == none
    assert repr(none) == 'none'
    assert str(none) == 'none'

def test_gamepadbutton_joystick():
    if False:
        while True:
            i = 10
    assert GamepadButton.trigger() == GamepadButton.joystick(0)
    assert GamepadButton.joystick(0).name == 'trigger'
    for i in range(1, 8):
        btn = GamepadButton.joystick(i)
        assert btn.name == 'joystick' + str(i + 1)

def test_keyboardbutton_ascii():
    if False:
        while True:
            i = 10
    assert KeyboardButton.space() == KeyboardButton.ascii_key(' ')
    assert KeyboardButton.backspace() == KeyboardButton.ascii_key('\x08')
    assert KeyboardButton.tab() == KeyboardButton.ascii_key('\t')
    assert KeyboardButton.enter() == KeyboardButton.ascii_key('\r')
    assert KeyboardButton.escape() == KeyboardButton.ascii_key('\x1b')
    assert KeyboardButton.ascii_key(' ').name == 'space'
    assert KeyboardButton.ascii_key('\x08').name == 'backspace'
    assert KeyboardButton.ascii_key('\t').name == 'tab'
    assert KeyboardButton.ascii_key('\r').name == 'enter'
    assert KeyboardButton.ascii_key('\x1b').name == 'escape'
    assert KeyboardButton.ascii_key('\x7f').name == 'delete'
    assert KeyboardButton.ascii_key('a').name == 'a'

def test_mousebutton():
    if False:
        i = 10
        return i + 15
    btns = [MouseButton.one(), MouseButton.two(), MouseButton.three(), MouseButton.four(), MouseButton.five()]
    for (i, btn) in enumerate(btns):
        assert MouseButton.button(i) == btn
        assert MouseButton.is_mouse_button(btn)
    assert MouseButton.button(5) == ButtonHandle.none()
    assert MouseButton.is_mouse_button(MouseButton.wheel_up())
    assert MouseButton.is_mouse_button(MouseButton.wheel_down())
    assert MouseButton.is_mouse_button(MouseButton.wheel_left())
    assert MouseButton.is_mouse_button(MouseButton.wheel_right())