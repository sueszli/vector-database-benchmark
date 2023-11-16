from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, Texture

def test_frame_empty():
    if False:
        for i in range(10):
            print('nop')
    frame = DirectFrame()
    assert not frame.hascomponent('text0')
    assert not frame.hascomponent('geom0')
    assert not frame.hascomponent('image0')

def test_frame_text():
    if False:
        for i in range(10):
            print('nop')
    frame = DirectFrame(text='Test')
    assert frame.hascomponent('text0')
    assert not frame.hascomponent('text1')
    assert frame.component('text0').text == 'Test'
    frame.setText('Foo')
    assert frame.component('text0').text == 'Foo'
    frame.setText(u'Foo')
    frame.clearText()
    assert not frame.hascomponent('text0')

def test_frame_text_states():
    if False:
        print('Hello World!')
    frame = DirectFrame(text=('A', 'B', 'C'), numStates=3)
    assert frame.hascomponent('text0')
    assert frame.hascomponent('text1')
    assert frame.hascomponent('text2')
    assert not frame.hascomponent('text3')
    assert frame.component('text0').text == 'A'
    assert frame.component('text1').text == 'B'
    assert frame.component('text2').text == 'C'
    frame.setText('Foo')
    assert frame.component('text0').text == 'Foo'
    assert frame.component('text1').text == 'Foo'
    assert frame.component('text2').text == 'Foo'
    frame.setText(('1', '2', '3'))
    assert frame.component('text0').text == '1'
    assert frame.component('text1').text == '2'
    assert frame.component('text2').text == '3'
    frame.setText(['1', '2', '3'])
    frame.clearText()
    assert not frame.hascomponent('text0')
    assert not frame.hascomponent('text1')
    assert not frame.hascomponent('text2')

def test_frame_geom():
    if False:
        for i in range(10):
            print('nop')
    frame = DirectFrame(geom=NodePath('geom-a'))
    assert frame.hascomponent('geom0')
    assert not frame.hascomponent('geom1')
    assert frame.component('geom0').name == 'geom-a'
    frame.setGeom(NodePath('geom-b'))
    assert frame.component('geom0').name == 'geom-b'
    frame.clearGeom()
    assert not frame.hascomponent('geom0')

def test_frame_geom_states():
    if False:
        while True:
            i = 10
    frame = DirectFrame(geom=(NodePath('A'), NodePath('B'), NodePath('C')), numStates=3)
    assert frame.hascomponent('geom0')
    assert frame.hascomponent('geom1')
    assert frame.hascomponent('geom2')
    assert not frame.hascomponent('geom3')
    assert frame.component('geom0').name == 'A'
    assert frame.component('geom1').name == 'B'
    assert frame.component('geom2').name == 'C'
    frame.setGeom(NodePath('Foo'))
    assert frame.component('geom0').name == 'Foo'
    assert frame.component('geom1').name == 'Foo'
    assert frame.component('geom2').name == 'Foo'
    states = (NodePath('1'), NodePath('2'), NodePath('3'))
    frame.setGeom(states)
    assert frame.component('geom0').name == '1'
    assert frame.component('geom1').name == '2'
    assert frame.component('geom2').name == '3'
    frame.setGeom(list(states))
    frame.clearGeom()
    assert not frame.hascomponent('geom0')
    assert not frame.hascomponent('geom1')
    assert not frame.hascomponent('geom2')