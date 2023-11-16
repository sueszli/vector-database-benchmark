from .helper import hopper

def test_sanity():
    if False:
        i = 10
        return i + 15
    im = hopper()
    type_repr = repr(type(im.getim()))
    assert 'PyCapsule' in type_repr
    assert isinstance(im.im.id, int)