import pytest
pytest.importorskip('panda3d.physics')
from panda3d.core import NodePath, PandaNode
from direct.particles.ParticleEffect import ParticleEffect
from direct.particles.Particles import Particles

def test_particle_birth_rate():
    if False:
        for i in range(10):
            print('nop')
    system = Particles('testSystem', 2)
    system.set_render_parent(NodePath(PandaNode('test')))
    system.set_spawn_render_node_path(NodePath(PandaNode('test')))
    assert system.get_birth_rate() == 0.5
    assert system.get_tics_since_birth() == 0
    assert system.get_living_particles() == 0
    system.update(0.6)
    assert system.get_living_particles() == 1
    system.update(0.5)
    assert system.get_living_particles() == 2
    system.update(0.5)
    assert system.get_living_particles() == 2

def test_particle_soft_start():
    if False:
        return 10
    effect = ParticleEffect()
    system = Particles('testSystem', 10)
    system.set_render_parent(NodePath(PandaNode('test')))
    system.set_spawn_render_node_path(NodePath(PandaNode('test')))
    effect.add_particles(system)
    system = effect.get_particles_list()[0]
    effect.soft_start()
    assert system.get_birth_rate() == 0.5
    system.soft_start(1)
    assert system.get_birth_rate() == 1
    effect.soft_start()
    assert system.get_tics_since_birth() == 0
    system.soft_start(br=-1, first_birth_delay=-2)
    assert system.get_birth_rate() == 1
    assert system.get_tics_since_birth() == 2
    effect.soft_start(firstBirthDelay=0.25)
    assert system.get_birth_rate() == 1
    assert system.get_tics_since_birth() == -0.25
    system.update(1)
    assert system.get_living_particles() == 0
    system.update(1)
    assert system.get_living_particles() == 1

def test_particle_burst_emission():
    if False:
        while True:
            i = 10
    effect = ParticleEffect()
    system = Particles('testSystem', 10)
    effect.add_particles(system)
    system.setRenderParent(NodePath(PandaNode('test')))
    system.setSpawnRenderNodePath(NodePath(PandaNode('test')))
    effect.softStop()
    assert system.getLivingParticles() == 0
    system.update(1)
    assert system.getLivingParticles() == 0
    effect.birthLitter()
    assert system.getLivingParticles() == 1
    effect.birth_litter()
    assert system.getLivingParticles() == 2