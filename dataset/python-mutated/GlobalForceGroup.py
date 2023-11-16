__all__ = ['GlobalForceGroup']
from . import ForceGroup
from direct.showbase.PhysicsManagerGlobal import physicsMgr

class GlobalForceGroup(ForceGroup.ForceGroup):

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        ForceGroup.ForceGroup.__init__(self, name)

    def addForce(self, force):
        if False:
            for i in range(10):
                print('nop')
        ForceGroup.ForceGroup.addForce(self, force)
        if force.isLinear():
            physicsMgr.addLinearForce(force)
        else:
            base.addAngularIntegrator()
            physicsMgr.addAngularForce(force)

    def removeForce(self, force):
        if False:
            print('Hello World!')
        ForceGroup.ForceGroup.removeForce(self, force)
        if force.isLinear():
            physicsMgr.removeLinearForce(force)
        else:
            physicsMgr.removeAngularForce(force)