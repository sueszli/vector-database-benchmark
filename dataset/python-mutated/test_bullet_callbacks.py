import pytest
from .conftest import simulate_until
bullet = pytest.importorskip('panda3d.bullet')
from panda3d import core
bullet_filter_algorithm = core.ConfigVariableString('bullet-filter-algorithm')

def test_tick(world):
    if False:
        i = 10
        return i + 15
    fired = []

    def callback(cd):
        if False:
            print('Hello World!')
        fired.append(isinstance(cd, bullet.BulletTickCallbackData))
    world.set_tick_callback(callback, False)
    assert fired == []
    world.do_physics(0.1)
    assert fired == [True]
    world.clear_tick_callback()
    world.do_physics(0.1)
    assert fired == [True]

@pytest.mark.skipif(bullet_filter_algorithm != 'callback', reason='bullet-filter-algorithm not set to callback')
def test_filter(world, scene):
    if False:
        i = 10
        return i + 15

    def callback(cd):
        if False:
            print('Hello World!')
        assert isinstance(cd, bullet.BulletFilterCallbackData)
        if {cd.node_0.name, cd.node_1.name} == {'ball', 'lower_box'}:
            cd.collide = False
        else:
            cd.collide = True
    world.set_filter_callback(callback)
    ball = scene.find('**/ball')
    assert simulate_until(world, lambda : ball.get_x() > 10)
    upper_box = scene.find('**/upper_box')
    assert not simulate_until(world, lambda : upper_box.get_z() < 5)

def test_contact(world, scene):
    if False:
        i = 10
        return i + 15
    contacts = []

    def callback(cd):
        if False:
            return 10
        assert isinstance(cd, bullet.BulletContactCallbackData)
        if {cd.node0.name, cd.node1.name} == {'upper_box', 'ramp'}:
            if not contacts:
                contacts.append(True)
    world.set_contact_added_callback(callback)
    ball = scene.find('**/ball')
    ramp = scene.find('**/ramp')
    ball.node().notify_collisions(True)
    ramp.node().notify_collisions(True)
    assert simulate_until(world, lambda : ball.get_x() > 0)
    assert simulate_until(world, lambda : bool(contacts))