from panda3d.core import NodePath
from panda3d.physics import ActorNode, AngularEulerIntegrator, ForceNode, LinearEulerIntegrator, LinearFrictionForce, LinearVectorForce

class FallTest(NodePath):

    def __init__(self):
        if False:
            print('Hello World!')
        NodePath.__init__(self, 'FallTest')

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.actorNode = ActorNode('FallTestActorNode')
        actorNodePath = self.attachNewNode(self.actorNode)
        avatarNodePath = base.loader.loadModel('models/misc/smiley')
        assert not avatarNodePath.isEmpty()
        avatarNodePath.reparentTo(actorNodePath)
        self.phys = base.physicsMgr
        if 1:
            fn = ForceNode('FallTest gravity')
            fnp = NodePath(fn)
            fnp.reparentTo(self)
            fnp.reparentTo(base.render)
            gravity = LinearVectorForce(0.0, 0.0, -0.5)
            fn.addForce(gravity)
            self.phys.addLinearForce(gravity)
            self.gravity = gravity
        if 0:
            fn = ForceNode('FallTest viscosity')
            fnp = NodePath(fn)
            fnp.reparentTo(self)
            fnp.reparentTo(base.render)
            self.avatarViscosity = LinearFrictionForce(0.0, 1.0, 0)
            fn.addForce(self.avatarViscosity)
            self.phys.addLinearForce(self.avatarViscosity)
        if 0:
            self.phys.attachLinearIntegrator(LinearEulerIntegrator())
        if 0:
            self.phys.attachAngularIntegrator(AngularEulerIntegrator())
        self.phys.attachPhysicalNode(self.actorNode)
        if 0:
            self.momentumForce = LinearVectorForce(0.0, 0.0, 0.0)
            fn = ForceNode('FallTest momentum')
            fnp = NodePath(fn)
            fnp.reparentTo(base.render)
            fn.addForce(self.momentumForce)
            self.phys.addLinearForce(self.momentumForce)
        if 0:
            self.acForce = LinearVectorForce(0.0, 0.0, 0.0)
            fn = ForceNode('FallTest avatarControls')
            fnp = NodePath(fn)
            fnp.reparentTo(base.render)
            fn.addForce(self.acForce)
            self.phys.addLinearForce(self.acForce)
        self.avatarNodePath = avatarNodePath

def test_FallTest(base):
    if False:
        return 10
    base.enableParticles()
    base.addAngularIntegrator()
    test = FallTest()
    test.reparentTo(base.render)
    test.setup()