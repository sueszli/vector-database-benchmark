from direct.distributed.DistributedSmoothNode import DistributedSmoothNode
from panda3d.core import NodePath
from direct.actor.Actor import Actor

class DistributedSmoothActor(DistributedSmoothNode, Actor):

    def __init__(self, cr):
        if False:
            print('Hello World!')
        Actor.__init__(self, 'models/ralph', {'run': 'models/ralph-run', 'walk': 'models/ralph-walk'})
        DistributedSmoothNode.__init__(self, cr)
        self.setCacheable(1)
        self.setScale(0.2)

    def generate(self):
        if False:
            return 10
        DistributedSmoothNode.generate(self)
        self.activateSmoothing(True, False)
        self.startSmooth()

    def announceGenerate(self):
        if False:
            for i in range(10):
                print('nop')
        DistributedSmoothNode.announceGenerate(self)
        self.reparentTo(render)

    def disable(self):
        if False:
            print('Hello World!')
        self.stopSmooth()
        if not self.isEmpty():
            Actor.unloadAnims(self, None, None, None)
        DistributedSmoothNode.disable(self)

    def delete(self):
        if False:
            print('Hello World!')
        try:
            self.DistributedActor_deleted
        except:
            self.DistributedActor_deleted = 1
            DistributedSmoothNode.delete(self)
            Actor.delete(self)

    def start(self):
        if False:
            return 10
        self.startPosHprBroadcast()

    def loop(self, animName):
        if False:
            i = 10
            return i + 15
        self.sendUpdate('loop', [animName])
        return Actor.loop(self, animName)

    def pose(self, animName, frame):
        if False:
            print('Hello World!')
        self.sendUpdate('pose', [animName, frame])
        return Actor.pose(self, animName, frame)