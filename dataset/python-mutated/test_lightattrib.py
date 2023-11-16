from panda3d import core
spot = core.NodePath(core.Spotlight('spot'))
point = core.NodePath(core.PointLight('point'))
ambient = core.NodePath(core.AmbientLight('ambient'))

def test_lightattrib_compose_add():
    if False:
        while True:
            i = 10
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(spot)
    lattr2 = core.LightAttrib.make()
    lattr2 = lattr2.add_on_light(point)
    lattr3 = lattr1.compose(lattr2)
    assert lattr3.get_num_on_lights() == 2
    assert spot in lattr3.on_lights
    assert point in lattr3.on_lights

def test_lightattrib_compose_subtract():
    if False:
        print('Hello World!')
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(spot)
    lattr1 = lattr1.add_on_light(point)
    lattr2 = core.LightAttrib.make()
    lattr2 = lattr2.add_off_light(ambient)
    lattr2 = lattr2.add_off_light(point)
    lattr3 = lattr1.compose(lattr2)
    assert lattr3.get_num_on_lights() == 1
    assert spot in lattr3.on_lights
    assert point not in lattr3.on_lights
    assert ambient not in lattr3.on_lights

def test_lightattrib_compose_both():
    if False:
        return 10
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(spot)
    lattr1 = lattr1.add_on_light(point)
    lattr2 = core.LightAttrib.make()
    lattr2 = lattr2.add_on_light(ambient)
    lattr2 = lattr2.add_on_light(spot)
    lattr2 = lattr2.add_off_light(point)
    lattr3 = lattr1.compose(lattr2)
    assert lattr3.get_num_on_lights() == 2
    assert spot in lattr3.on_lights
    assert point not in lattr3.on_lights
    assert ambient in lattr3.on_lights

def test_lightattrib_compose_alloff():
    if False:
        while True:
            i = 10
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(spot)
    lattr1 = lattr1.add_on_light(point)
    assert lattr1.get_num_on_lights() == 2
    lattr2 = core.LightAttrib.make_all_off()
    assert lattr2.has_all_off()
    lattr3 = lattr1.compose(lattr2)
    assert lattr3.get_num_on_lights() == 0
    assert lattr3.get_num_off_lights() == 0
    assert lattr3.has_all_off()

def test_lightattrib_compare():
    if False:
        while True:
            i = 10
    lattr1 = core.LightAttrib.make()
    lattr2 = core.LightAttrib.make()
    assert lattr1.compare_to(lattr2) == 0
    lattr2 = core.LightAttrib.make_all_off()
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(spot)
    lattr2 = core.LightAttrib.make()
    lattr2 = lattr2.add_on_light(spot)
    assert lattr1.compare_to(lattr2) == 0
    assert lattr2.compare_to(lattr1) == 0
    lattr2 = lattr2.add_on_light(point)
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make()
    lattr1 = lattr1.add_on_light(point)
    lattr2 = core.LightAttrib.make()
    lattr2 = lattr2.add_on_light(spot)
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make().add_on_light(spot)
    lattr2 = core.LightAttrib.make().add_off_light(spot)
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make().add_off_light(spot)
    lattr2 = core.LightAttrib.make().add_off_light(spot)
    assert lattr1.compare_to(lattr2) == 0
    assert lattr2.compare_to(lattr1) == 0
    lattr1 = core.LightAttrib.make().add_off_light(spot)
    lattr2 = core.LightAttrib.make_all_off()
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make().add_off_light(spot)
    lattr2 = core.LightAttrib.make_all_off()
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)
    lattr1 = core.LightAttrib.make().add_off_light(spot)
    lattr2 = core.LightAttrib.make().add_off_light(point)
    assert lattr1.compare_to(lattr2) != 0
    assert lattr2.compare_to(lattr1) != 0
    assert lattr2.compare_to(lattr1) == -lattr1.compare_to(lattr2)