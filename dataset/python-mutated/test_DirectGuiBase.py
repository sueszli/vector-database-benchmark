from direct.gui.DirectGuiBase import DirectGuiBase, DirectGuiWidget, toggleGuiGridSnap, setGuiGridSpacing
from direct.gui.OnscreenText import OnscreenText
from direct.gui import DirectGuiGlobals as DGG
from direct.showbase.ShowBase import ShowBase
from direct.showbase import ShowBaseGlobal
from direct.showbase.MessengerGlobal import messenger
from panda3d import core
import pytest

def test_create_DirectGuiBase():
    if False:
        while True:
            i = 10
    baseitem = DirectGuiBase()

def test_defineoptions():
    if False:
        print('Hello World!')
    baseitem = DirectGuiBase()
    testoptiondefs = (('test', 0, None),)
    baseitem.defineoptions({}, testoptiondefs)
    assert baseitem['test'] == 0

def test_addoptions():
    if False:
        print('Hello World!')
    baseitem = DirectGuiBase()
    testoptiondefs = (('test', 0, None),)
    baseitem._optionInfo = {}
    baseitem._constructorKeywords = {}
    baseitem.addoptions(testoptiondefs, {})
    assert baseitem['test'] == 0

def test_initialiseoptions():
    if False:
        i = 10
        return i + 15
    baseitem = DirectGuiBase()

    class testWidget(DirectGuiBase):

        def __init__(self):
            if False:
                return 10
            pass
    tw = testWidget()
    baseitem.initialiseoptions(tw)
    testoptiondefs = (('test', 0, None),)
    tw.defineoptions({}, testoptiondefs)
    baseitem.initialiseoptions(tw)
    assert tw['test'] == 0

def test_postInitialiseFunc():
    if False:
        print('Hello World!')
    baseitem = DirectGuiBase()

    def func_a():
        if False:
            i = 10
            return i + 15
        global test_value_a
        test_value_a = 1

    def func_b():
        if False:
            print('Hello World!')
        global test_value_b
        test_value_b = 1
    baseitem.postInitialiseFuncList.append(func_a)
    baseitem.postInitialiseFuncList.append(func_b)
    baseitem.postInitialiseFunc()
    global test_value_a
    assert test_value_a == 1
    global test_value_b
    assert test_value_b == 1

def test_isinitoption():
    if False:
        while True:
            i = 10
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-true', 0, DGG.INITOPT), ('test-false', 0, None))
    baseitem.defineoptions({}, testoptiondefs)
    assert baseitem.isinitoption('test-true') == True
    assert baseitem.isinitoption('test-false') == False

def test_options():
    if False:
        return 10
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-1', 0, DGG.INITOPT), ('test-2', 0, None))
    baseitem.defineoptions({}, testoptiondefs)
    assert len(baseitem.options()) == 2

def test_get_options():
    if False:
        while True:
            i = 10
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-1', 0, None),)
    baseitem.defineoptions({}, testoptiondefs)
    assert baseitem.configure() == {'test-1': ('test-1', 0, 0)}
    assert baseitem.configure('test-1') == ('test-1', 0, 0)
    assert baseitem['test-1'] == 0
    assert baseitem.cget('test-1') == 0

def test_set_options():
    if False:
        print('Hello World!')
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-1', 0, DGG.INITOPT), ('test-2', 0, None))
    baseitem.defineoptions({}, testoptiondefs)
    baseitem.configure('test-1', **{'test-1': 1})
    assert baseitem['test-1'] == 0
    baseitem['test-1'] = 1
    assert baseitem['test-1'] == 0
    baseitem.configure('test-2', **{'test-2': 2})
    assert baseitem['test-2'] == 2
    baseitem['test-2'] = 1
    assert baseitem['test-2'] == 1

def test_component_handling():
    if False:
        return 10
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-1', 0, None),)
    baseitem.defineoptions({}, testoptiondefs)
    assert len(baseitem.components()) == 0
    baseitem.createcomponent('componentName', (), 'componentGroup', OnscreenText, (), text='Test', parent=core.NodePath())
    assert len(baseitem.components()) == 1
    assert baseitem.hascomponent('componentName')
    component = baseitem.component('componentName')
    assert component.text == 'Test'
    baseitem.destroycomponent('componentName')
    assert baseitem.hascomponent('componentName') == False

def test_destroy():
    if False:
        while True:
            i = 10
    baseitem = DirectGuiBase()
    testoptiondefs = (('test-1', 0, None),)
    baseitem.defineoptions({}, testoptiondefs)
    baseitem.destroy()
    assert not hasattr(baseitem, '_optionInfo')
    assert not hasattr(baseitem, '__componentInfo')
    assert not hasattr(baseitem, 'postInitialiseFuncList')

def test_bindings():
    if False:
        return 10
    baseitem = DirectGuiBase()
    global commandCalled
    commandCalled = False

    def command(**kw):
        if False:
            i = 10
            return i + 15
        global commandCalled
        commandCalled = True
        assert True
    baseitem.bind(DGG.B1CLICK, command)
    messenger.send(DGG.B1CLICK + baseitem.guiId)
    assert commandCalled
    baseitem.unbind(DGG.B1CLICK)
    commandCalled = False
    messenger.send(DGG.B1CLICK + baseitem.guiId)
    assert not commandCalled

def test_toggle_snap():
    if False:
        print('Hello World!')
    try:
        DirectGuiWidget.snapToGrid = 0
        item = DirectGuiWidget()
        assert item.snapToGrid == 0
        toggleGuiGridSnap()
        assert item.snapToGrid == 1
    finally:
        DirectGuiWidget.snapToGrid = 0

def test_toggle_spacing():
    if False:
        return 10
    try:
        DirectGuiWidget.gridSpacing = 0
        item = DirectGuiWidget()
        setGuiGridSpacing(5)
        assert item.gridSpacing == 5
    finally:
        DirectGuiWidget.gridSpacing = 0.05

@pytest.mark.skipif(not ShowBaseGlobal.__dev__, reason='requires want-dev')
def test_track_gui_items():
    if False:
        for i in range(10):
            print('nop')
    page = core.load_prc_file_data('', 'track-gui-items true')
    try:
        item = DirectGuiWidget()
        id = item.guiId
        assert id in ShowBase.guiItems
        assert ShowBase.guiItems[id] == item
        item.destroy()
        assert id not in ShowBase.guiItems
    finally:
        core.unload_prc_file(page)