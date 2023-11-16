import sys
from direct.showbase.ShowBase import ShowBase
from direct.showbase.InputStateGlobal import inputState
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import LVector3
from panda3d.core import LPoint3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import GeomNode
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletSoftBodyNode
from panda3d.bullet import BulletSoftBodyConfig

class Game(ShowBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        ShowBase.__init__(self)
        base.set_background_color(0.1, 0.1, 0.8, 1)
        base.set_frame_rate_meter(True)
        base.cam.set_pos(0, -60, 20)
        base.cam.look_at(0, 0, 0)
        alight = AmbientLight('ambientLight')
        alight.set_color((0.5, 0.5, 0.5, 1))
        alightNP = render.attach_new_node(alight)
        dlight = DirectionalLight('directionalLight')
        dlight.set_direction((1, 1, -1))
        dlight.set_color((0.7, 0.7, 0.7, 1))
        dlightNP = render.attach_new_node(dlight)
        render.clear_light()
        render.set_light(alightNP)
        render.set_light(dlightNP)
        self.accept('escape', self.do_exit)
        self.accept('r', self.do_reset)
        self.accept('f1', base.toggle_wireframe)
        self.accept('f2', base.toggle_texture)
        self.accept('f3', self.toggle_debug)
        self.accept('f5', self.do_screenshot)
        taskMgr.add(self.update, 'updateWorld')
        self.setup()

    def do_exit(self):
        if False:
            i = 10
            return i + 15
        self.cleanup()
        sys.exit(1)

    def do_reset(self):
        if False:
            while True:
                i = 10
        self.cleanup()
        self.setup()

    def toggle_debug(self):
        if False:
            while True:
                i = 10
        if self.debugNP.is_hidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def do_screenshot(self):
        if False:
            for i in range(10):
                print('nop')
        base.screenshot('Bullet')

    def update(self, task):
        if False:
            while True:
                i = 10
        dt = globalClock.get_dt()
        self.world.do_physics(dt, 10, 0.008)
        return task.cont

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        self.world = None
        self.worldNP.remove_node()

    def setup(self):
        if False:
            return 10
        self.worldNP = render.attach_new_node('World')
        self.debugNP = self.worldNP.attach_new_node(BulletDebugNode('Debug'))
        self.debugNP.show()
        self.world = BulletWorld()
        self.world.set_gravity((0, 0, -9.81))
        self.world.set_debug_node(self.debugNP.node())
        p0 = LPoint3(-20, -20, 0)
        p1 = LPoint3(-20, 20, 0)
        p2 = LPoint3(20, -20, 0)
        p3 = LPoint3(20, 20, 0)
        mesh = BulletTriangleMesh()
        mesh.add_triangle(p0, p1, p2)
        mesh.add_triangle(p1, p2, p3)
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        np = self.worldNP.attach_new_node(BulletRigidBodyNode('Mesh'))
        np.node().add_shape(shape)
        np.set_pos(0, 0, -2)
        np.set_collide_mask(BitMask32.all_on())
        self.world.attach(np.node())
        origin = LPoint3(0, 0, 0)
        size = LVector3(2, 10, 1)
        shape = BulletBoxShape(size * 0.5)
        for i in range(10):
            pos = origin + size * i
            pos.setY(0)
            np = self.worldNP.attach_new_node(BulletRigidBodyNode('Stair{}'.format(i)))
            np.node().add_shape(shape)
            np.set_pos(pos)
            np.set_collide_mask(BitMask32.all_on())
            npV = loader.load_model('models/box.egg')
            npV.reparent_to(np)
            npV.set_scale(size)
            self.world.attach(np.node())
        info = self.world.get_world_info()
        info.set_air_density(1.2)
        info.set_water_density(0)
        info.set_water_offset(0)
        info.set_water_normal((0, 0, 0))
        center = LPoint3(0, 0, 0)
        radius = LVector3(1, 1, 1) * 1.5
        node = BulletSoftBodyNode.make_ellipsoid(info, center, radius, 128)
        node.set_name('Ellipsoid')
        node.get_material(0).set_linear_stiffness(0.1)
        node.get_cfg().set_dynamic_friction_coefficient(1)
        node.get_cfg().set_damping_coefficient(0.001)
        node.get_cfg().set_pressure_coefficient(1500)
        node.set_total_mass(30, True)
        node.set_pose(True, False)
        np = self.worldNP.attach_new_node(node)
        np.set_pos(15, 0, 12)
        self.world.attach(np.node())
        geom = BulletHelper.make_geom_from_faces(node)
        node.link_geom(geom)
        nodeV = GeomNode('EllipsoidVisual')
        nodeV.add_geom(geom)
        npV = np.attach_new_node(nodeV)
game = Game()
game.run()