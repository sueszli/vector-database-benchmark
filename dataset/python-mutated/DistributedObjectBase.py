from direct.showbase.DirectObject import DirectObject
from direct.directnotify.DirectNotifyGlobal import directNotify

class DistributedObjectBase(DirectObject):
    """
    The Distributed Object class is the base class for all network based
    (i.e. distributed) objects.  These will usually (always?) have a
    dclass entry in a \\*.dc file.
    """
    notify = directNotify.newCategory('DistributedObjectBase')

    def __init__(self, cr):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        self.cr = cr
        self.parentId = None
        self.zoneId = None
    if __debug__:

        def status(self, indent=0):
            if False:
                return 10
            '\n            print out "doId(parentId, zoneId) className"\n            '
            spaces = ' ' * (indent + 2)
            try:
                print('%s%s:' % (' ' * indent, self.__class__.__name__))
                print('%sfrom DistributedObject doId:%s, parent:%s, zone:%s' % (spaces, self.doId, self.parentId, self.zoneId))
            except Exception as e:
                print('%serror printing status %s' % (spaces, e))

    def getLocation(self):
        if False:
            i = 10
            return i + 15
        try:
            if self.parentId == 0 and self.zoneId == 0:
                return None
            if self.parentId == 4294967295 and self.zoneId == 4294967295:
                return None
            return (self.parentId, self.zoneId)
        except AttributeError:
            return None

    def handleChildArrive(self, childObj, zoneId):
        if False:
            print('Hello World!')
        '\n        A new child has just setLocation beneath us.  Give us a\n        chance to run code when a new child sets location to us. For\n        example, we may want to scene graph reparent the child to\n        some subnode we own.\n        '
        assert self.notify.debugCall()

    def handleChildArriveZone(self, childObj, zoneId):
        if False:
            while True:
                i = 10
        '\n        A child has just changed zones beneath us with setLocation.\n        Give us a chance to run code when an existing child sets\n        location to us. For example, we may want to scene graph\n        reparent the child to some subnode we own.\n        '
        assert self.notify.debugCall()

    def handleChildLeave(self, childObj, zoneId):
        if False:
            while True:
                i = 10
        '\n        A child is about to setLocation away from us.  Give us a\n        chance to run code just before a child sets location away from us.\n        '
        assert self.notify.debugCall()

    def handleChildLeaveZone(self, childObj, zoneId):
        if False:
            while True:
                i = 10
        '\n        A child is about to setLocation to another zone beneath us.\n        Give us a chance to run code just before a child sets\n        location to that zone.\n        '
        assert self.notify.debugCall()

    def handleQueryObjectChildrenLocalDone(self, context):
        if False:
            while True:
                i = 10
        assert self.notify.debugCall()

    def getParentObj(self):
        if False:
            print('Hello World!')
        if self.parentId is None:
            return None
        return self.cr.doId2do.get(self.parentId)

    def hasParentingRules(self):
        if False:
            print('Hello World!')
        return self.dclass.getFieldByName('setParentingRules') is not None

    def delete(self):
        if False:
            return 10
        '\n        Override this to handle cleanup right before this object\n        gets deleted.\n        '