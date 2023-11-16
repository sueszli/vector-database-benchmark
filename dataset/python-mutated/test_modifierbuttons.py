from panda3d.core import ModifierButtons

def test_modifierbuttons_empty():
    if False:
        for i in range(10):
            print('nop')
    btns = ModifierButtons()
    assert btns == ModifierButtons(btns)
    assert btns != ModifierButtons()
    assert btns.matches(ModifierButtons())
    assert not btns.is_down('alt')
    assert not btns.is_any_down()
    assert not btns.has_button('alt')
    assert btns.get_prefix() == ''
    assert btns.get_num_buttons() == 0
    assert len(btns.buttons) == 0

def test_modifierbuttons_cow():
    if False:
        return 10
    btns1 = ModifierButtons()
    btns1.add_button('space')
    btns2 = ModifierButtons(btns1)
    assert tuple(btns2.buttons) == tuple(btns1.buttons)
    btns1.add_button('enter')
    assert tuple(btns1.buttons) == ('space', 'enter')
    assert tuple(btns2.buttons) == ('space',)
    btns3 = ModifierButtons(btns2)
    assert tuple(btns3.buttons) == tuple(btns2.buttons)
    btns3.add_button('escape')
    assert tuple(btns2.buttons) == ('space',)
    assert tuple(btns3.buttons) == ('space', 'escape')

def test_modifierbuttons_assign():
    if False:
        i = 10
        return i + 15
    btns1 = ModifierButtons()
    btns1.add_button('space')
    btns2 = ModifierButtons()
    btns2.assign(btns1)
    assert btns1 == btns2
    assert tuple(btns1.buttons) == tuple(btns2.buttons)

def test_modifierbuttons_state():
    if False:
        print('Hello World!')
    btns = ModifierButtons()
    btns.add_button('alt')
    btns.add_button('shift')
    btns.add_button('control')
    assert not btns.is_any_down()
    btns.button_down('enter')
    assert not btns.is_any_down()
    btns.button_down('shift')
    assert btns.is_any_down()
    assert not btns.is_down(0)
    assert btns.is_down(1)
    assert not btns.is_down(2)
    btns.button_up('shift')
    assert not btns.is_any_down()
    assert not btns.is_down(0)
    assert not btns.is_down(1)
    assert not btns.is_down(2)
    btns.button_down('alt')
    btns.button_down('shift')
    assert btns.is_any_down()
    assert btns.is_down(0)
    assert btns.is_down(1)
    assert not btns.is_down(2)
    btns.all_buttons_up()
    assert not btns.is_any_down()
    assert not btns.is_down(0)
    assert not btns.is_down(1)
    assert not btns.is_down(2)