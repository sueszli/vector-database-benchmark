from hypothesis.utils.dynamicvariables import DynamicVariable

def test_can_assign():
    if False:
        print('Hello World!')
    d = DynamicVariable(1)
    assert d.value == 1
    with d.with_value(2):
        assert d.value == 2
    assert d.value == 1

def test_can_nest():
    if False:
        while True:
            i = 10
    d = DynamicVariable(1)
    with d.with_value(2):
        assert d.value == 2
        with d.with_value(3):
            assert d.value == 3
        assert d.value == 2
    assert d.value == 1