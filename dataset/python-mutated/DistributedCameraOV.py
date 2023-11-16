from direct.distributed.DistributedObjectOV import DistributedObjectOV

class DistributedCameraOV(DistributedObjectOV):

    def __init__(self, cr):
        if False:
            print('Hello World!')
        DistributedObjectOV.__init__(self, cr)
        self.parent = 0
        self.fixtures = []
        self.accept('refresh-fixture', self.refreshFixture)

    def delete(self):
        if False:
            i = 10
            return i + 15
        self.ignore('escape')
        self.ignore('refresh-fixture')
        DistributedObjectOV.delete(self)

    def getObject(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cr.getDo(self.getDoId())

    def setCamParent(self, doId):
        if False:
            print('Hello World!')
        self.parent = doId

    def setFixtures(self, fixtures):
        if False:
            i = 10
            return i + 15
        self.fixtures = fixtures

    def storeToFile(self, name):
        if False:
            return 10
        f = open('cameras-%s.txt' % name, 'w')
        f.writelines(self.getObject().pack())
        f.close()

    def unpackFixture(self, data):
        if False:
            while True:
                i = 10
        data = data.strip().replace('Camera', '')
        (pos, hpr, fov) = eval(data)
        return (pos, hpr, fov)

    def loadFromFile(self, name):
        if False:
            i = 10
            return i + 15
        self.b_setFixtures([])
        f = open('cameras-%s.txt' % name, 'r')
        for line in f.readlines():
            (pos, hpr, fov) = self.unpackFixture(line)
            self.addFixture([pos[0], pos[1], pos[2], hpr[0], hpr[1], hpr[2], fov[0], fov[1], 'Standby'])
        f.close()

    def refreshFixture(self, id, data):
        if False:
            print('Hello World!')
        (pos, hpr, fov) = self.unpackFixture(data)
        fixture = self.fixtures[id]
        fixture = [pos[0], pos[1], pos[2], hpr[0], hpr[1], hpr[2], fov[0], fov[1], fixture[8]]
        self.d_setFixtures(self.fixtures)

    def b_setFixtures(self, fixtures):
        if False:
            i = 10
            return i + 15
        self.getObject().setFixtures(fixtures)
        self.setFixtures(fixtures)
        self.d_setFixtures(fixtures)

    def d_setFixtures(self, fixtures):
        if False:
            while True:
                i = 10
        self.sendUpdate('setFixtures', [fixtures])

    def addFixture(self, fixture, index=None):
        if False:
            return 10
        if index is not None:
            self.fixtures.insert(index, fixture)
        else:
            self.fixtures.append(fixture)
        self.b_setFixtures(self.fixtures)
        return self.fixtures.index(fixture)

    def blinkFixture(self, index):
        if False:
            print('Hello World!')
        if index < len(self.fixtures):
            fixture = self.fixtures[index]
            fixture[6] = 'Blinking'
            self.b_setFixtures(self.fixtures)

    def standbyFixture(self, index):
        if False:
            print('Hello World!')
        if index < len(self.fixtures):
            fixture = self.fixtures[index]
            fixture[6] = 'Standby'
            self.b_setFixtures(self.fixtures)

    def testFixture(self, index):
        if False:
            i = 10
            return i + 15
        if index < len(self.fixtures):
            self.getObject().testFixture(index)

    def removeFixture(self, index):
        if False:
            print('Hello World!')
        self.fixtures.pop(index)
        self.b_setFixtures(self.fixtures)

    def saveFixture(self, index=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Position the camera with ~oobe, then call this to save its telemetry.\n        '
        parent = self.getObject().getCamParent()
        pos = base.cam.getPos(parent)
        hpr = base.cam.getHpr(parent)
        return self.addFixture([pos[0], pos[1], pos[2], hpr[0], hpr[1], hpr[2], 'Standby'], index)

    def startRecording(self):
        if False:
            while True:
                i = 10
        self.accept('escape', self.stopRecording)
        for fixture in self.fixtures:
            fixture[6] = 'Recording'
        self.b_setFixtures(self.fixtures)

    def stopRecording(self):
        if False:
            while True:
                i = 10
        self.ignore('escape')
        for fixture in self.fixtures:
            fixture[6] = 'Standby'
        self.b_setFixtures(self.fixtures)