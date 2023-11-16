from panda3d import core
stage1 = core.TextureStage('stage1')
stage2 = core.TextureStage('stage2')
stage3 = core.TextureStage('stage3')
tex1 = core.Texture('tex1')
tex2 = core.Texture('tex2')
tex3 = core.Texture('tex3')

def test_textureattrib_compose_empty():
    if False:
        i = 10
        return i + 15
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 1
    assert stage1 in tattr3.on_stages

def test_textureattrib_compose_add():
    if False:
        i = 10
        return i + 15
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage2, tex2)
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 2
    assert stage1 in tattr3.on_stages
    assert stage2 in tattr3.on_stages

def test_textureattrib_compose_override():
    if False:
        for i in range(10):
            print('nop')
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage1, tex2)
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 1
    assert stage1 in tattr3.on_stages
    assert tattr3.get_on_texture(stage1) == tex2

def test_textureattrib_compose_subtract():
    if False:
        print('Hello World!')
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr1 = tattr1.add_on_stage(stage2, tex2)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_off_stage(stage3)
    tattr2 = tattr2.add_off_stage(stage2)
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 1
    assert stage1 in tattr3.on_stages
    assert stage2 not in tattr3.on_stages
    assert stage3 not in tattr3.on_stages

def test_textureattrib_compose_both():
    if False:
        i = 10
        return i + 15
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr1 = tattr1.add_on_stage(stage2, tex2)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage3, tex3)
    tattr2 = tattr2.add_on_stage(stage1, tex1)
    tattr2 = tattr2.add_off_stage(stage2)
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 2
    assert stage1 in tattr3.on_stages
    assert stage2 not in tattr3.on_stages
    assert stage3 in tattr3.on_stages

def test_textureattrib_compose_alloff():
    if False:
        i = 10
        return i + 15
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr1 = tattr1.add_on_stage(stage2, tex2)
    assert tattr1.get_num_on_stages() == 2
    tattr2 = core.TextureAttrib.make_all_off()
    assert tattr2.has_all_off()
    tattr3 = tattr1.compose(tattr2)
    assert tattr3.get_num_on_stages() == 0
    assert tattr3.get_num_off_stages() == 0
    assert tattr3.has_all_off()

def test_textureattrib_implicit_sort():
    if False:
        return 10
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr1 = tattr1.add_on_stage(stage2, tex2)
    assert tattr1.get_on_stage(0) == stage1
    assert tattr1.get_on_stage(1) == stage2
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage2, tex2)
    tattr2 = tattr2.add_on_stage(stage1, tex1)
    assert tattr2.get_on_stage(0) == stage2
    assert tattr2.get_on_stage(1) == stage1
    assert tattr1.compare_to(tattr2) == -tattr2.compare_to(tattr1)

def test_textureattrib_replace():
    if False:
        for i in range(10):
            print('nop')
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = tattr1.add_on_stage(stage1, tex1)
    assert tattr1.get_num_on_stages() == 1
    assert tattr2.get_num_on_stages() == 1
    assert tattr1.compare_to(tattr2) == 0

def test_textureattrib_compare():
    if False:
        i = 10
        return i + 15
    tattr1 = core.TextureAttrib.make()
    tattr2 = core.TextureAttrib.make()
    assert tattr1.compare_to(tattr2) == 0
    tattr2 = core.TextureAttrib.make_all_off()
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make()
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_off_stage(stage1)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make_all_off()
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_off_stage(stage1)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make_all_off()
    tattr1 = tattr2.add_off_stage(stage1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_off_stage(stage2)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage1, tex2)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage2, tex2)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage1, tex1)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage1, tex1)
    assert tattr1.compare_to(tattr2) == 0
    assert tattr2.compare_to(tattr1) == 0
    tattr2 = tattr2.add_on_stage(stage2, tex2)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)
    tattr1 = core.TextureAttrib.make()
    tattr1 = tattr1.add_on_stage(stage2, tex2)
    tattr2 = core.TextureAttrib.make()
    tattr2 = tattr2.add_on_stage(stage1, tex1)
    assert tattr1.compare_to(tattr2) != 0
    assert tattr2.compare_to(tattr1) != 0
    assert tattr2.compare_to(tattr1) == -tattr1.compare_to(tattr2)