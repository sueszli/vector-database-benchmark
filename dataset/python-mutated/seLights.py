from direct.showbase.DirectObject import *
from direct.directtools import DirectUtil
from panda3d.core import *

class seLight(NodePath):

    def __init__(self, light, parent, type, lightcolor=VBase4(0.3, 0.3, 0.3, 1), specularColor=VBase4(1), position=Point3(0, 0, 0), orientation=Vec3(1, 0, 0), constant=1.0, linear=0.0, quadratic=0.0, exponent=0.0, tag='', lence=None):
        if False:
            while True:
                i = 10
        NodePath.__init__(self)
        self.light = light
        self.type = type
        self.lightcolor = lightcolor
        self.specularColor = specularColor
        self.position = position
        self.orientation = orientation
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.exponent = exponent
        self.lence = lence
        self.active = True
        self.LightNode = parent.attachNewNode(light)
        self.LightNode.setTag('Metadata', tag)
        if self.type == 'spot':
            self.LightNode.setHpr(self.orientation)
            self.LightNode.setPos(self.position)
        else:
            self.LightNode.setHpr(self.orientation)
            self.LightNode.setPos(self.position)
        self.assign(self.LightNode)
        if self.type == 'spot':
            self.helpModel = loader.loadModel('models/misc/Spotlight')
        elif self.type == 'point':
            self.helpModel = loader.loadModel('models/misc/Pointlight')
        elif self.type == 'directional':
            self.helpModel = loader.loadModel('models/misc/Dirlight')
        else:
            self.helpModel = loader.loadModel('models/misc/sphere')
        self.helpModel.setColor(self.lightcolor)
        self.helpModel.reparentTo(self)
        DirectUtil.useDirectRenderStyle(self.helpModel)
        if not (self.type == 'directional' or self.type == 'point' or self.type == 'spot'):
            self.helpModel.hide()

    def getLight(self):
        if False:
            print('Hello World!')
        return self.light

    def getLightColor(self):
        if False:
            print('Hello World!')
        return self.lightcolor

    def getName(self):
        if False:
            print('Hello World!')
        return self.light.getName()

    def rename(self, name):
        if False:
            while True:
                i = 10
        self.light.setName(name)
        self.setName(name)

    def getType(self):
        if False:
            return 10
        return self.type

    def setColor(self, color):
        if False:
            return 10
        self.light.setColor(color)
        self.lightcolor = color
        if self.type == 'directional' or self.type == 'point':
            self.helpModel.setColor(self.lightcolor)
        return

    def getSpecColor(self):
        if False:
            return 10
        return self.specularColor

    def setSpecColor(self, color):
        if False:
            while True:
                i = 10
        self.light.setSpecularColor(color)
        self.specularcolor = color
        return

    def getPosition(self):
        if False:
            i = 10
            return i + 15
        self.position = self.LightNode.getPos()
        return self.position

    def setPosition(self, pos):
        if False:
            i = 10
            return i + 15
        self.LightNode.setPos(pos)
        self.position = pos
        return

    def getOrientation(self):
        if False:
            i = 10
            return i + 15
        self.orientation = self.LightNode.getHpr()
        return self.orientation

    def setOrientation(self, orient):
        if False:
            while True:
                i = 10
        self.LightNode.setHpr(orient)
        self.orientation = orient
        return

    def getAttenuation(self):
        if False:
            return 10
        return Vec3(self.constant, self.linear, self.quadratic)

    def setConstantAttenuation(self, value):
        if False:
            i = 10
            return i + 15
        self.light.setAttenuation(Vec3(value, self.linear, self.quadratic))
        self.constant = value
        return

    def setLinearAttenuation(self, value):
        if False:
            while True:
                i = 10
        self.light.setAttenuation(Vec3(self.constant, value, self.quadratic))
        self.linear = value
        return

    def setQuadraticAttenuation(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.light.setAttenuation(Vec3(self.constant, self.linear, value))
        self.quadratic = value
        return

    def getExponent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.exponent

    def setExponent(self, value):
        if False:
            print('Hello World!')
        self.light.setExponent(value)
        self.exponent = value
        return

class seLightManager(NodePath):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        NodePath.__init__(self)
        self.lnode = render.attachNewNode('Lights')
        self.assign(self.lnode)
        self.lightAttrib = LightAttrib.makeAllOff()
        self.lightDict = {}
        self.ambientCount = 0
        self.directionalCount = 0
        self.pointCount = 0
        self.spotCount = 0
        self.helpModel = loader.loadModel('models/misc/sphere')
        self.helpModel.reparentTo(self)
        self.helpModel.hide()

    def create(self, type='ambient', lightcolor=VBase4(0.3, 0.3, 0.3, 1), specularColor=VBase4(1), position=Point3(0, 0, 0), orientation=Vec3(1, 0, 0), constant=1.0, linear=0.0, quadratic=0.0, exponent=0.0, tag='', name='DEFAULT_NAME'):
        if False:
            while True:
                i = 10
        lence = None
        if type == 'ambient':
            self.ambientCount += 1
            if name == 'DEFAULT_NAME':
                light = AmbientLight('ambient_' + repr(self.ambientCount))
            else:
                light = AmbientLight(name)
            light.setColor(lightcolor)
        elif type == 'directional':
            self.directionalCount += 1
            if name == 'DEFAULT_NAME':
                light = DirectionalLight('directional_' + repr(self.directionalCount))
            else:
                light = DirectionalLight(name)
            light.setColor(lightcolor)
            light.setSpecularColor(specularColor)
        elif type == 'point':
            self.pointCount += 1
            if name == 'DEFAULT_NAME':
                light = PointLight('point_' + repr(self.pointCount))
            else:
                light = PointLight(name)
            light.setColor(lightcolor)
            light.setSpecularColor(specularColor)
            light.setAttenuation(Vec3(constant, linear, quadratic))
        elif type == 'spot':
            self.spotCount += 1
            if name == 'DEFAULT_NAME':
                light = Spotlight('spot_' + repr(self.spotCount))
            else:
                light = Spotlight(name)
            light.setColor(lightcolor)
            lence = PerspectiveLens()
            light.setLens(lence)
            light.setSpecularColor(specularColor)
            light.setAttenuation(Vec3(constant, linear, quadratic))
            light.setExponent(exponent)
        else:
            print('Invalid light type')
            return None
        lightNode = seLight(light, self, type, lightcolor=lightcolor, specularColor=specularColor, position=position, orientation=orientation, constant=constant, linear=linear, quadratic=quadratic, exponent=exponent, tag=tag, lence=lence)
        self.lightDict[light.getName()] = lightNode
        self.setOn(lightNode)
        return (self.lightDict.keys(), lightNode)

    def addLight(self, light):
        if False:
            while True:
                i = 10
        type = light.getType().getName().lower()
        specularColor = VBase4(1)
        position = Point3(0, 0, 0)
        orientation = Vec3(1, 0, 0)
        constant = 1.0
        linear = 0.0
        quadratic = 0.0
        exponent = 0.0
        lence = None
        lightcolor = light.getColor()
        if type == 'ambientlight':
            type = 'ambient'
            self.ambientCount += 1
        elif type == 'directionallight':
            type = 'directional'
            self.directionalCount += 1
            orientation = light.getDirection()
            position = light.getPoint()
            specularColor = light.getSpecularColor()
        elif type == 'pointlight':
            type = 'point'
            self.pointCount += 1
            position = light.getPoint()
            specularColor = light.getSpecularColor()
            Attenuation = light.getAttenuation()
            constant = Attenuation.getX()
            linear = Attenuation.getY()
            quadratic = Attenuation.getZ()
        elif type == 'spotlight':
            type = 'spot'
            self.spotCount += 1
            specularColor = light.getSpecularColor()
            Attenuation = light.getAttenuation()
            constant = Attenuation.getX()
            linear = Attenuation.getY()
            quadratic = Attenuation.getZ()
            exponent = light.getExponent()
        else:
            print('Invalid light type')
            return None
        lightNode = seLight(light, self, type, lightcolor=lightcolor, specularColor=specularColor, position=position, orientation=orientation, constant=constant, linear=linear, quadratic=quadratic, exponent=exponent, lence=lence)
        self.lightDict[light.getName()] = lightNode
        self.setOn(lightNode)
        return (self.lightDict.keys(), lightNode)

    def delete(self, name, removeEntry=True):
        if False:
            i = 10
            return i + 15
        type = self.lightDict[name].getType()
        if type == 'ambient':
            self.ambientCount -= 1
        elif type == 'directional':
            self.directionalCount -= 1
        elif type == 'point':
            self.pointCount -= 1
        elif type == 'spot':
            self.spotCount -= 1
        self.setOff(self.lightDict[name])
        self.lightDict[name].removeChildren()
        self.lightDict[name].removeNode()
        if removeEntry:
            del self.lightDict[name]
        return self.lightDict.keys()

    def deleteAll(self):
        if False:
            print('Hello World!')
        for name in self.lightDict:
            self.delete(name, removeEntry=False)
        self.lightDict.clear()

    def isLight(self, name):
        if False:
            for i in range(10):
                print('nop')
        return name in self.lightDict

    def rename(self, oName, nName):
        if False:
            return 10
        if self.isLight(oName):
            lightNode = self.lightDict[oName]
            self.lightDict[nName] = lightNode
            lightNode.rename(nName)
            del self.lightDict[oName]
            return (self.lightDict.keys(), lightNode)
        else:
            print('----Light Mnager: No such Light!')

    def getLightNodeList(self):
        if False:
            for i in range(10):
                print('nop')
        list = []
        for name in self.lightDict:
            list.append(self.lightDict[name])
        return list

    def getLightNodeDict(self):
        if False:
            while True:
                i = 10
        return self.lightDict

    def getLightList(self):
        if False:
            i = 10
            return i + 15
        list = []
        for name in self.lightDict:
            list.append(name)
        return list

    def getLightNode(self, lightName):
        if False:
            for i in range(10):
                print('nop')
        if lightName in self.lightDict:
            return self.lightDict[lightName]

    def allOn(self):
        if False:
            i = 10
            return i + 15
        render.node().setAttrib(self.lightAttrib)
        render.setMaterial(Material())

    def allOff(self):
        if False:
            for i in range(10):
                print('nop')
        render.node().clearAttrib(LightAttrib.getClassType())

    def toggle(self):
        if False:
            i = 10
            return i + 15
        if render.node().hasAttrib(LightAttrib.getClassType()):
            self.allOff()
        else:
            self.allOn()

    def setOn(self, lightNode):
        if False:
            return 10
        self.lightAttrib = self.lightAttrib.addLight(lightNode.getLight())
        lightNode.active = True
        if render.node().hasAttrib(LightAttrib.getClassType()):
            render.node().setAttrib(self.lightAttrib)

    def setOff(self, lightNode):
        if False:
            return 10
        lightNode.active = False
        self.lightAttrib = self.lightAttrib.removeLight(lightNode.getLight())
        if render.node().hasAttrib(LightAttrib.getClassType()):
            render.node().setAttrib(self.lightAttrib)

    def getList(self):
        if False:
            print('Hello World!')
        return self.lightDict.keys()