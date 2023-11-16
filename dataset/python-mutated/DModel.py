from direct.distributed.DistributedNode import DistributedNode

class DModel(DistributedNode):

    def __init__(self, cr):
        if False:
            for i in range(10):
                print('nop')
        DistributedNode.__init__(self, cr)
        self.model = loader.loadModel('smiley.egg')
        self.model.reparentTo(self)

    def announceGenerate(self):
        if False:
            i = 10
            return i + 15
        ' This method is called after generate(), after all of the\n        required fields have been filled in.  At the time of this call,\n        the distributed object is ready for use. '
        DistributedNode.announceGenerate(self)
        self.reparentTo(render)

    def disable(self):
        if False:
            return 10
        ' This method is called when the object is removed from the\n        scene, for instance because it left the zone.  It is balanced\n        against generate(): for each generate(), there will be a\n        corresponding disable().  Everything that was done in\n        generate() or announceGenerate() should be undone in disable().\n\n        After a disable(), the object might be cached in memory in case\n        it will eventually reappear.  The DistributedObject should be\n        prepared to receive another generate() for an object that has\n        already received disable().\n\n        Note that the above is only strictly true for *cacheable*\n        objects.  Most objects are, by default, non-cacheable; you\n        have to call obj.setCacheable(True) (usually in the\n        constructor) to make it cacheable.  Until you do this, your\n        non-cacheable object will always receive a delete() whenever\n        it receives a disable(), and it will never be stored in a\n        cache.\n        '
        self.detachNode()
        DistributedNode.disable(self)

    def delete(self):
        if False:
            while True:
                i = 10
        ' This method is called after disable() when the object is to\n        be completely removed, for instance because the other user\n        logged off.  We will not expect to see this object again; it\n        will not be cached.  This is stronger than disable(), and the\n        object may remove any structures it needs to in order to allow\n        it to be completely deleted from memory.  This balances against\n        __init__(): every DistributedObject that is created will\n        eventually get delete() called for it exactly once. '
        self.model = None
        DistributedNode.delete(self)