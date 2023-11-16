from panda3d.core import AmbientLight, DirectionalLight, LightAttrib, Material, NodePath, PerspectiveLens, PointLight, Spotlight, VBase4
from direct.showbase.MessengerGlobal import messenger

class DirectLight(NodePath):

    def __init__(self, light, parent):
        if False:
            while True:
                i = 10
        NodePath.__init__(self)
        self.light = light
        self.assign(parent.attachNewNode(self.light))

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.light.getName()

    def getLight(self):
        if False:
            while True:
                i = 10
        return self.light

class DirectLights(NodePath):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        if parent is None:
            parent = base.render
        NodePath.__init__(self)
        self.assign(parent.attachNewNode('DIRECT Lights'))
        self.lightDict = {}
        self.ambientCount = 0
        self.directionalCount = 0
        self.pointCount = 0
        self.spotCount = 0

    def __getitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.lightDict.get(name, None)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.lightDict)

    def delete(self, light):
        if False:
            i = 10
            return i + 15
        del self.lightDict[light.getName()]
        self.setOff(light)
        light.removeNode()

    def deleteAll(self):
        if False:
            while True:
                i = 10
        for light in self:
            self.delete(light)

    def asList(self):
        if False:
            return 10
        return [self[n] for n in self.getNameList()]

    def getNameList(self):
        if False:
            for i in range(10):
                print('nop')
        return sorted((x.getName() for x in self.lightDict.values()))

    def create(self, ltype):
        if False:
            return 10
        ltype = ltype.lower()
        if ltype == 'ambient':
            self.ambientCount += 1
            light = AmbientLight('ambient-' + repr(self.ambientCount))
            light.setColor(VBase4(0.3, 0.3, 0.3, 1))
        elif ltype == 'directional':
            self.directionalCount += 1
            light = DirectionalLight('directional-' + repr(self.directionalCount))
            light.setColor(VBase4(1))
        elif ltype == 'point':
            self.pointCount += 1
            light = PointLight('point-' + repr(self.pointCount))
            light.setColor(VBase4(1))
        elif ltype == 'spot':
            self.spotCount += 1
            light = Spotlight('spot-' + repr(self.spotCount))
            light.setColor(VBase4(1))
            light.setLens(PerspectiveLens())
        else:
            print('Invalid light type')
            return None
        directLight = DirectLight(light, self)
        self.lightDict[directLight.getName()] = directLight
        self.setOn(directLight)
        messenger.send('DIRECT_addLight', [directLight])
        return directLight

    def createDefaultLights(self):
        if False:
            return 10
        self.create('ambient')
        self.create('directional')

    def allOn(self):
        if False:
            while True:
                i = 10
        '\n        Turn on all DIRECT lights\n        '
        for light in self.lightDict.values():
            self.setOn(light)
        render.setMaterial(Material())

    def allOff(self):
        if False:
            i = 10
            return i + 15
        '\n        Turn off all DIRECT lights\n        '
        for light in self.lightDict.values():
            self.setOff(light)

    def toggle(self):
        if False:
            i = 10
            return i + 15
        "\n        Toggles light attribute, but doesn't toggle individual lights\n        "
        if render.node().hasAttrib(LightAttrib.getClassType()):
            self.allOff()
        else:
            self.allOn()

    def setOn(self, directLight):
        if False:
            while True:
                i = 10
        '\n        Turn on the given directLight\n        '
        render.setLight(directLight)

    def setOff(self, directLight):
        if False:
            for i in range(10):
                print('nop')
        '\n        Turn off the given directLight\n        '
        render.clearLight(directLight)