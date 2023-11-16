import sys
from direct.showbase.ShowBase import ShowBase
from direct.showbase.InputStateGlobal import inputState
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import LPoint3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletSphereShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletSphericalConstraint
from panda3d.bullet import BulletDebugNode

class Game(ShowBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        ShowBase.__init__(self)
        base.set_background_color(0.1, 0.1, 0.8, 1)
        base.set_frame_rate_meter(True)
        base.cam.set_pos(0, -20, 5)
        base.cam.look_at(0, 0, 5)
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
        self.accept('enter', self.do_shoot)
        taskMgr.add(self.update, 'updateWorld')
        self.setup()

    def do_exit(self):
        if False:
            return 10
        self.cleanup()
        sys.exit(1)

    def do_reset(self):
        if False:
            return 10
        self.cleanup()
        self.setup()

    def toggle_debug(self):
        if False:
            print('Hello World!')
        if self.debugNP.is_hidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def do_screenshot(self):
        if False:
            for i in range(10):
                print('nop')
        base.screenshot('Bullet')

    def do_shoot(self):
        if False:
            for i in range(10):
                print('nop')
        pMouse = base.mouseWatcherNode.get_mouse()
        pFrom = LPoint3()
        pTo = LPoint3()
        base.camLens.extrude(pMouse, pFrom, pTo)
        pFrom = render.get_relative_point(base.cam, pFrom)
        pTo = render.get_relative_point(base.cam, pTo)
        v = pTo - pFrom
        v.normalize()
        v *= 100.0
        shape = BulletSphereShape(0.3)
        body = BulletRigidBodyNode('Bullet')
        bodyNP = self.worldNP.attach_new_node(body)
        bodyNP.node().add_shape(shape)
        bodyNP.node().set_mass(1.0)
        bodyNP.node().set_linear_velocity(v)
        bodyNP.node().set_ccd_motion_threshold(1e-07)
        bodyNP.node().set_ccd_swept_sphere_radius(0.5)
        bodyNP.set_collide_mask(BitMask32.all_on())
        bodyNP.set_pos(pFrom)
        visNP = loader.load_model('models/ball.egg')
        visNP.set_scale(0.8)
        visNP.reparent_to(bodyNP)
        self.world.attach(bodyNP.node())
        taskMgr.do_method_later(2, self.do_remove, 'doRemove', extraArgs=[bodyNP], appendTask=True)

    def do_remove(self, bodyNP, task):
        if False:
            print('Hello World!')
        self.world.remove(bodyNP.node())
        bodyNP.remove_node()
        return task.done

    def update(self, task):
        if False:
            for i in range(10):
                print('nop')
        dt = globalClock.get_dt()
        self.world.do_physics(dt, 20, 1.0 / 180.0)
        return task.cont

    def cleanup(self):
        if False:
            while True:
                i = 10
        self.worldNP.remove_node()
        self.worldNP = None
        self.world = None

    def setup(self):
        if False:
            while True:
                i = 10
        self.worldNP = render.attach_new_node('World')
        self.debugNP = self.worldNP.attach_new_node(BulletDebugNode('Debug'))
        self.debugNP.show()
        self.debugNP.node().show_wireframe(True)
        self.debugNP.node().show_constraints(True)
        self.debugNP.node().show_bounding_boxes(False)
        self.debugNP.node().show_normals(False)
        self.world = BulletWorld()
        self.world.set_gravity((0, 0, -9.81))
        self.world.set_debug_node(self.debugNP.node())
        shape = BulletBoxShape((0.5, 0.5, 0.5))
        bodyA = BulletRigidBodyNode('Box A')
        bodyNP = self.worldNP.attach_new_node(bodyA)
        bodyNP.node().add_shape(shape)
        bodyNP.set_collide_mask(BitMask32.all_on())
        bodyNP.set_pos(-1, 0, 4)
        visNP = loader.load_model('models/box.egg')
        visNP.clear_model_nodes()
        visNP.reparent_to(bodyNP)
        self.world.attach(bodyA)
        shape = BulletBoxShape((0.5, 0.5, 0.5))
        bodyB = BulletRigidBodyNode('Box B')
        bodyNP = self.worldNP.attach_new_node(bodyB)
        bodyNP.node().add_shape(shape)
        bodyNP.node().set_mass(1.0)
        bodyNP.node().set_deactivation_enabled(False)
        bodyNP.node().setLinearDamping(0.6)
        bodyNP.node().setAngularDamping(0.6)
        bodyNP.set_collide_mask(BitMask32.all_on())
        bodyNP.set_pos(2, 0, 0)
        visNP = loader.load_model('models/box.egg')
        visNP.clear_model_nodes()
        visNP.reparent_to(bodyNP)
        self.world.attach(bodyB)
        pivotA = (2, 0, 0)
        pivotB = (0, 0, 4)
        joint = BulletSphericalConstraint(bodyA, bodyB, pivotA, pivotB)
        joint.set_debug_draw_size(2.0)
        self.world.attach(joint)
game = Game()
game.run()