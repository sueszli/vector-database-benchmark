import pytest
from .conftest import simulate_until
bullet = pytest.importorskip('panda3d.bullet')
from panda3d import core

def test_basics(world, scene):
    if False:
        i = 10
        return i + 15
    ball = scene.find('**/ball')
    assert simulate_until(world, lambda : ball.get_x() >= 0)
    upper_box = scene.find('**/upper_box')
    assert upper_box.get_z() > 5.0
    assert simulate_until(world, lambda : upper_box.get_z() < 5.0)

def test_restitution(world, scene):
    if False:
        return 10
    ball = scene.find('**/ball')
    scene.find('**/ramp').node().restitution = 1.0
    for with_bounce in (False, True):
        ball.node().set_angular_velocity(core.Vec3(0))
        ball.node().set_linear_velocity(core.Vec3(0))
        ball.set_pos(-2, 0, 100)
        ball.node().restitution = 1.0 * with_bounce
        assert simulate_until(world, lambda : ball.get_x() >= 0)
        if with_bounce:
            assert ball.get_z() > 1.2
        else:
            assert ball.get_z() < 1.2

def test_friction(world, scene):
    if False:
        print('Hello World!')
    ball = scene.find('**/ball')
    for with_friction in (False, True):
        ball.node().set_angular_velocity(core.Vec3(-1000, 0, 0))
        ball.node().set_linear_velocity(core.Vec3(0))
        ball.set_pos(-2, 0, 5)
        ball.node().friction = 1.0 * with_friction
        assert simulate_until(world, lambda : ball.get_x() >= 0)
        if with_friction:
            assert ball.get_y() > 1
        else:
            assert abs(ball.get_y()) < 0.1