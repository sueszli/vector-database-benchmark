import pytest
from pytest import approx
bullet = pytest.importorskip('panda3d.bullet')
from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletVehicle

def test_get_steering():
    if False:
        while True:
            i = 10
    world = BulletWorld()
    shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
    body = BulletRigidBodyNode('Vehicle')
    body.addShape(shape)
    world.attach(body)
    vehicle = BulletVehicle(world, body)
    world.attachVehicle(vehicle)
    wheel = vehicle.createWheel()
    wheel.setSteering(30.0)
    assert wheel.getSteering() == approx(30.0)