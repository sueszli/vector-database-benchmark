"""DistributedObjectGlobal module: contains the DistributedObjectGlobal class"""
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DistributedObject import DistributedObject

class DistributedObjectGlobal(DistributedObject):
    """
    The Distributed Object Global class is the base class for global
    network based (i.e. distributed) objects.
    """
    notify = directNotify.newCategory('DistributedObjectGlobal')
    neverDisable = 1

    def __init__(self, cr):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        DistributedObject.__init__(self, cr)
        self.parentId = 0
        self.zoneId = 0