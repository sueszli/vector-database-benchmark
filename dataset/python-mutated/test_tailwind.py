from nicegui import Tailwind, ui
from .screen import Screen

def test_tailwind_builder(screen: Screen):
    if False:
        i = 10
        return i + 15
    ui.label('A').tailwind('bg-red-500', 'text-white')
    screen.open('/')
    assert screen.find('A').get_attribute('class') == 'bg-red-500 text-white'

def test_tailwind_call(screen: Screen):
    if False:
        for i in range(10):
            print('nop')
    ui.label('A').tailwind('bg-red-500 text-white')
    screen.open('/')
    assert screen.find('A').get_attribute('class') == 'bg-red-500 text-white'

def test_tailwind_apply(screen: Screen):
    if False:
        return 10
    style = Tailwind().background_color('red-500').text_color('white')
    ui.label('A').tailwind(style)
    b = ui.label('B')
    style.apply(b)
    screen.open('/')
    assert screen.find('A').get_attribute('class') == 'bg-red-500 text-white'
    assert screen.find('B').get_attribute('class') == 'bg-red-500 text-white'

def test_empty_values():
    if False:
        return 10
    label = ui.label('A')
    label.tailwind.border_width('')
    assert 'border' in label._classes