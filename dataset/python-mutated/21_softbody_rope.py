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
from panda3d.core import RopeNode
from panda3d.core import NurbsCurveEvaluator
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletSoftBodyNode
from panda3d.bullet import BulletSoftBodyConfig

class Game(ShowBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        ShowBase.__init__(self)
        base.set_background_color(0.1, 0.1, 0.8, 1)
        base.set_frame_rate_meter(True)
        base.cam.set_pos(0, -40, 10)
        base.cam.look_at(0, 0, 0)
        alight = AmbientLight('ambientLight')
        alight.set_color((0.5, 0.5, 0.5, 1))
        alightNP = render.attach_new_node(alight)
        dlight = DirectionalLight('directionalLight')
        dlight.set_direction((5, 0, -2))
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
            for i in range(10):
                print('nop')
        self.cleanup()
        self.setup()

    def toggle_debug(self):
        if False:
            for i in range(10):
                print('nop')
        if self.debugNP.is_hidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def do_screenshot(self):
        if False:
            print('Hello World!')
        base.screenshot('Bullet')

    def update(self, task):
        if False:
            i = 10
            return i + 15
        dt = globalClock.get_dt()
        self.world.do_physics(dt, 10, 0.004)
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
        info = self.world.get_world_info()
        info.set_air_density(1.2)
        info.set_water_density(0)
        info.set_water_offset(0)
        info.set_water_normal((0, 0, 0))

        def make(p1):
            if False:
                print('Hello World!')
            n = 8
            p2 = p1 + LVector3(10, 0, 0)
            bodyNode = BulletSoftBodyNode.makeRope(info, p1, p2, n, 1)
            bodyNode.set_total_mass(50.0)
            bodyNP = self.worldNP.attach_new_node(bodyNode)
            self.world.attach(bodyNode)
            curve = NurbsCurveEvaluator()
            curve.reset(n + 2)
            bodyNode.link_curve(curve)
            visNode = RopeNode('')
            visNode.set_curve(curve)
            visNode.set_render_mode(RopeNode.RM_tube)
            visNode.set_uv_mode(RopeNode.UV_parametric)
            visNode.set_num_subdiv(4)
            visNode.set_num_slices(8)
            visNode.set_thickness(0.4)
            visNP = self.worldNP.attach_new_node(visNode)
            visNP.set_texture(loader.load_texture('models/maps/color-grid.rgb'))
            return bodyNP
        np1 = make(LPoint3(-2, -1, 8))
        np2 = make(LPoint3(-2, 1, 8))
        shape = BulletBoxShape((2, 2, 6))
        boxNP = self.worldNP.attach_new_node(BulletRigidBodyNode('Box'))
        boxNP.node().set_mass(50.0)
        boxNP.node().add_shape(shape)
        boxNP.set_pos(10, 0, 8)
        boxNP.set_collide_mask(BitMask32.all_on())
        self.world.attach(boxNP.node())
        np1.node().append_anchor(np1.node().get_num_nodes() - 1, boxNP.node())
        np2.node().append_anchor(np1.node().get_num_nodes() - 1, boxNP.node())
        visNP = loader.load_model('models/box.egg')
        visNP.clear_model_nodes()
        visNP.set_scale(4, 4, 12)
        visNP.reparent_to(boxNP)
game = Game()
game.run()