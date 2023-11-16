"""DistributedActor module: contains the DistributedActor class"""
__all__ = ['DistributedActor']
from direct.distributed import DistributedNode
from . import Actor

class DistributedActor(DistributedNode.DistributedNode, Actor.Actor):

    def __init__(self, cr):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, 'DistributedActor_initialized'):
            self.DistributedActor_initialized = 1
            Actor.Actor.__init__(self)
            DistributedNode.DistributedNode.__init__(self, cr)
            self.setCacheable(1)

    def disable(self):
        if False:
            return 10
        if not self.isEmpty():
            Actor.Actor.unloadAnims(self, None, None, None)
        DistributedNode.DistributedNode.disable(self)

    def delete(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'DistributedActor_deleted'):
            self.DistributedActor_deleted = 1
            DistributedNode.DistributedNode.delete(self)
            Actor.Actor.delete(self)

    def loop(self, animName, restart=1, partName=None, fromFrame=None, toFrame=None):
        if False:
            i = 10
            return i + 15
        return Actor.Actor.loop(self, animName, restart, partName, fromFrame, toFrame)