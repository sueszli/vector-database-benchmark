from cura.Scene.GCodeListDecorator import GCodeListDecorator

def test_setAndGetList():
    if False:
        i = 10
        return i + 15
    decorator = GCodeListDecorator()
    decorator.setGCodeList(['Test'])
    assert decorator.getGCodeList() == ['Test']

def test_copyGCodeDecorator():
    if False:
        while True:
            i = 10
    decorator = GCodeListDecorator()
    decorator.setGCodeList(['Test'])
    import copy
    copied_decorator = copy.deepcopy(decorator)
    assert decorator.getGCodeList() == copied_decorator.getGCodeList()