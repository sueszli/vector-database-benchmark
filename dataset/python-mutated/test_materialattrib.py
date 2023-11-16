from panda3d import core

def test_materialattrib_compare():
    if False:
        i = 10
        return i + 15
    mat1 = core.Material()
    mat2 = core.Material()
    mattr1 = core.MaterialAttrib.make_off()
    mattr2 = core.MaterialAttrib.make_off()
    assert mattr1.compare_to(mattr2) == 0
    assert mattr2.compare_to(mattr1) == 0
    mattr1 = core.MaterialAttrib.make_off()
    mattr2 = core.MaterialAttrib.make(mat1)
    assert mattr1 != mattr2
    assert mattr1.compare_to(mattr2) != 0
    assert mattr2.compare_to(mattr1) != 0
    assert mattr1.compare_to(mattr2) == -mattr2.compare_to(mattr1)
    mattr1 = core.MaterialAttrib.make(mat1)
    mattr2 = core.MaterialAttrib.make(mat1)
    assert mattr1.compare_to(mattr2) == 0
    assert mattr2.compare_to(mattr1) == 0
    mattr1 = core.MaterialAttrib.make(mat1)
    mattr2 = core.MaterialAttrib.make(mat2)
    assert mattr1 != mattr2
    assert mattr1.compare_to(mattr2) != 0
    assert mattr2.compare_to(mattr1) != 0
    assert mattr1.compare_to(mattr2) == -mattr2.compare_to(mattr1)