"""ProjectileInterval module: contains the ProjectileInterval class"""
__all__ = ['ProjectileInterval']
from panda3d.core import CollisionParabola, LParabola, NodePath, Point3, VBase3
from direct.directnotify.DirectNotifyGlobal import directNotify
from .Interval import Interval
from direct.showbase import PythonUtil

class ProjectileInterval(Interval):
    """ProjectileInterval class: moves a nodepath through the trajectory
    of a projectile under the influence of gravity"""
    notify = directNotify.newCategory('ProjectileInterval')
    projectileIntervalNum = 1
    gravity = 32.0

    def __init__(self, node, startPos=None, endPos=None, duration=None, startVel=None, endZ=None, wayPoint=None, timeToWayPoint=None, gravityMult=None, name=None, collNode=None):
        if False:
            return 10
        "\n        You may specify several different sets of input parameters.\n        (If startPos is not provided, it will be obtained from the node's\n        position at the time that the interval is first started. Note that\n        in this case you must provide a duration of some kind.)\n\n        # go from startPos to endPos in duration seconds\n        startPos, endPos, duration\n        # given a starting velocity, go for a specific time period\n        startPos, startVel, duration\n        # given a starting velocity, go until you hit a given Z plane\n        startPos, startVel, endZ\n        # pass through wayPoint at time 'timeToWayPoint'. Go until\n        # you hit a given Z plane\n        startPos, wayPoint, timeToWayPoint, endZ\n\n        You may alter gravity by providing a multiplier in 'gravityMult'.\n        '2.' will make gravity twice as strong, '.5' half as strong.\n        '-1.' will reverse gravity\n\n        If collNode is not None, it should be an empty CollisionNode\n        which will be filled with an appropriate CollisionParabola\n        when the interval starts.  This CollisionParabola will be set\n        to match the interval's parabola, and its t1, t2 values will\n        be updated automatically as the interval plays.  It will *not*\n        be automatically removed from the node when the interval\n        finishes.\n\n        "
        self.node = node
        self.collNode = collNode
        if self.collNode:
            if isinstance(self.collNode, NodePath):
                self.collNode = self.collNode.node()
            assert self.collNode.getNumSolids() == 0
        if name is None:
            name = '%s-%s' % (self.__class__.__name__, self.projectileIntervalNum)
            ProjectileInterval.projectileIntervalNum += 1
        args = (startPos, endPos, duration, startVel, endZ, wayPoint, timeToWayPoint, gravityMult)
        self.implicitStartPos = 0
        if startPos is None:
            if duration is None:
                self.notify.error('must provide either startPos or duration')
            self.duration = duration
            self.trajectoryArgs = args
            self.implicitStartPos = 1
        else:
            self.trajectoryArgs = args
            self.__calcTrajectory(*args)
        Interval.__init__(self, name, self.duration)

    def __calcTrajectory(self, startPos=None, endPos=None, duration=None, startVel=None, endZ=None, wayPoint=None, timeToWayPoint=None, gravityMult=None):
        if False:
            return 10
        if startPos is None:
            startPos = self.node.getPos()

        def doIndirections(*items):
            if False:
                for i in range(10):
                    print('nop')
            result = []
            for item in items:
                if callable(item):
                    item = item()
                result.append(item)
            return result
        (startPos, endPos, startVel, endZ, gravityMult, wayPoint, timeToWayPoint) = doIndirections(startPos, endPos, startVel, endZ, gravityMult, wayPoint, timeToWayPoint)
        self.startPos = startPos
        self.zAcc = -self.gravity
        if gravityMult:
            self.zAcc *= gravityMult

        def calcStartVel(startPos, endPos, duration, zAccel):
            if False:
                return 10
            if duration == 0:
                return Point3(0, 0, 0)
            else:
                return Point3((endPos[0] - startPos[0]) / duration, (endPos[1] - startPos[1]) / duration, (endPos[2] - startPos[2] - 0.5 * zAccel * duration * duration) / duration)

        def calcTimeOfImpactOnPlane(startHeight, endHeight, startVel, accel):
            if False:
                print('Hello World!')
            return PythonUtil.solveQuadratic(accel * 0.5, startVel, startHeight - endHeight)

        def calcTimeOfLastImpactOnPlane(startHeight, endHeight, startVel, accel):
            if False:
                while True:
                    i = 10
            time = calcTimeOfImpactOnPlane(startHeight, endHeight, startVel, accel)
            if not time:
                return None
            if isinstance(time, list):
                assert self.notify.debug('projectile hits plane twice at times: %s' % time)
                time = max(*time)
            else:
                assert self.notify.debug('projectile hits plane once at time: %s' % time)
            return time
        if None not in (endPos, duration):
            assert not startVel
            assert not endZ
            assert not wayPoint
            assert not timeToWayPoint
            self.duration = duration
            self.endPos = endPos
            self.startVel = calcStartVel(self.startPos, self.endPos, self.duration, self.zAcc)
        elif None not in (startVel, duration):
            assert not endPos
            assert not endZ
            assert not wayPoint
            assert not timeToWayPoint
            self.duration = duration
            self.startVel = startVel
            self.endPos = None
        elif None not in (startVel, endZ):
            assert not endPos
            assert not duration
            assert not wayPoint
            assert not timeToWayPoint
            self.startVel = startVel
            time = calcTimeOfLastImpactOnPlane(self.startPos[2], endZ, self.startVel[2], self.zAcc)
            if time is None:
                self.notify.error('projectile never reaches plane Z=%s' % endZ)
            self.duration = time
            self.endPos = None
        elif None not in (wayPoint, timeToWayPoint, endZ):
            assert not endPos
            assert not duration
            assert not startVel
            self.startVel = calcStartVel(self.startPos, wayPoint, timeToWayPoint, self.zAcc)
            time = calcTimeOfLastImpactOnPlane(self.startPos[2], endZ, self.startVel[2], self.zAcc)
            if time is None:
                self.notify.error('projectile never reaches plane Z=%s' % endZ)
            self.duration = time
            self.endPos = None
        else:
            self.notify.error('invalid set of inputs to ProjectileInterval')
        self.parabola = LParabola(VBase3(0, 0, 0.5 * self.zAcc), self.startVel, self.startPos)
        if not self.endPos:
            self.endPos = self.__calcPos(self.duration)
        assert self.notify.debug('startPos: %s' % repr(self.startPos))
        assert self.notify.debug('endPos:   %s' % repr(self.endPos))
        assert self.notify.debug('duration: %s' % self.duration)
        assert self.notify.debug('startVel: %s' % repr(self.startVel))
        assert self.notify.debug('z-accel:  %s' % self.zAcc)

    def __initialize(self):
        if False:
            print('Hello World!')
        if self.implicitStartPos:
            self.__calcTrajectory(*self.trajectoryArgs)

    def testTrajectory(self):
        if False:
            print('Hello World!')
        try:
            self.__calcTrajectory(*self.trajectoryArgs)
        except Exception:
            assert self.notify.error('invalid projectile parameters')
            return False
        return True

    def privInitialize(self, t):
        if False:
            print('Hello World!')
        self.__initialize()
        if self.collNode:
            self.collNode.clearSolids()
            csolid = CollisionParabola(self.parabola, 0, 0)
            self.collNode.addSolid(csolid)
        Interval.privInitialize(self, t)

    def privInstant(self):
        if False:
            for i in range(10):
                print('nop')
        self.__initialize()
        Interval.privInstant(self)
        if self.collNode:
            self.collNode.clearSolids()
            csolid = CollisionParabola(self.parabola, 0, self.duration)
            self.collNode.addSolid(csolid)

    def __calcPos(self, t):
        if False:
            return 10
        return self.parabola.calcPoint(t)

    def privStep(self, t):
        if False:
            while True:
                i = 10
        self.node.setFluidPos(self.__calcPos(t))
        Interval.privStep(self, t)
        if self.collNode and self.collNode.getNumSolids() > 0:
            csolid = self.collNode.modifySolid(0)
            csolid.setT1(csolid.getT2())
            csolid.setT2(t)