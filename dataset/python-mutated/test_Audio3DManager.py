from direct.showbase.Audio3DManager import Audio3DManager
from panda3d import core
import pytest

@pytest.fixture(scope='session')
def manager3d():
    if False:
        while True:
            i = 10
    root = core.NodePath('root')
    manager = core.AudioManager.create_AudioManager()
    manager3d = Audio3DManager(manager, root=root)
    yield manager3d
    del manager3d
    manager.shutdown()

def test_audio3dmanager_velocity(manager3d):
    if False:
        for i in range(10):
            print('nop')
    sound = manager3d.load_sfx('nonexistent')
    object = core.NodePath('object')
    object.set_pos(0, 0, 0)
    object.set_fluid_pos(1, 2, 3)
    assert object.get_pos_delta() == (1, 2, 3)
    res = manager3d.attach_sound_to_object(sound, object)
    assert res
    clock = core.ClockObject.get_global_clock()
    clock.mode = core.ClockObject.M_slave
    clock.dt = 0.5
    manager3d.set_sound_velocity_auto(sound)
    assert manager3d.get_sound_velocity(sound) == (2, 4, 6)
    manager3d.set_sound_velocity(sound, (5, 5, 5))
    assert manager3d.get_sound_velocity(sound) == (5, 5, 5)

def test_audio3dmanager_weak(manager3d):
    if False:
        for i in range(10):
            print('nop')
    sound = manager3d.load_sfx('nonexistent')
    object = core.NodePath('object')
    res = manager3d.attach_sound_to_object(sound, object)
    assert res
    assert object in manager3d.sound_dict
    manager3d.update()
    assert object in manager3d.sound_dict
    object.remove_node()
    manager3d.update()
    assert object not in manager3d.sound_dict