from direct.showbase.TkGlobal import *
import Pmw
from direct.tkwidgets import Dial
from direct.tkwidgets import Floater
from tkinter.filedialog import askopenfilename
from seLights import *
from seFileSaver import *
from direct.actor import Actor
import os
import sys
import seParticleEffect
import seParticles

class dataHolder:
    ModelDic = {}
    ModelRefDic = {}
    ActorDic = {}
    ActorRefDic = {}
    curveDict = {}
    collisionDict = {}
    blendAnimDict = {}
    collisionVisable = True
    dummyDict = {}
    particleDict = {}
    particleNodes = {}
    ModelNum = 0
    ActorNum = 0
    theScene = None
    CollisionHandler = CollisionHandlerEvent()
    controlType = 'Keyboard'
    controlTarget = camera
    keyboardMapDict = {'KeyForward': 'arrow_up', 'KeyBackward': 'arrow_down', 'KeyLeft': 'arrow_left', 'KeyRight': 'arrow_right', 'KeyUp': '', 'KeyDown': '', 'KeyTurnRight': '', 'KeyTurnLeft': '', 'KeyTurnUp': '', 'KeyTurnDown': '', 'KeyRollRight': '', 'KeyRollLeft': '', 'KeyScaleUp': '', 'KeyScaleDown': '', 'KeyScaleXUp': '', 'KeyScaleXDown': '', 'KeyScaleYUp': '', 'KeyScaleYDown': '', 'KeyScaleZUp': '', 'KeyScaleZDown': ''}
    keyboardSpeedDict = {'SpeedForward': 0, 'SpeedBackward': 0, 'SpeedLeft': 0, 'SpeedRight': 0, 'SpeedUp': 0, 'SpeedDown': 0, 'SpeedTurnRight': 0, 'SpeedTurnLeft': 0, 'SpeedTurnUp': 0, 'SpeedTurnDown': 0, 'SpeedRollRight': 0, 'SpeedRollLeft': 0, 'SpeedScaleUp': 0, 'SpeedScaleDown': 0, 'SpeedScaleXUp': 0, 'SpeedScaleXDown': 0, 'SpeedScaleYUp': 0, 'SpeedScaleYDown': 0, 'SpeedScaleZUp': 0, 'SpeedScaleZDown': 0}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.lightManager = seLightManager()
        self.lightManager.allOn()
        self.CollisionHandler.setInPattern('%fnenter%in')
        self.CollisionHandler.setOutPattern('%fnexit%in')
        pass

    def resetAll(self):
        if False:
            while True:
                i = 10
        for index in self.ModelDic:
            self.ModelDic[index].removeNode()
        for index in self.ActorDic:
            self.ActorDic[index].removeNode()
        for index in self.dummyDict:
            self.dummyDict[index].removeNode()
        for index in self.collisionDict:
            self.collisionDict[index].removeNode()
        for index in self.particleNodes:
            self.particleDict[index].cleanup()
            self.particleNodes[index].removeNode()
        self.ModelDic.clear()
        self.ModelRefDic.clear()
        self.ActorDic.clear()
        self.ActorRefDic.clear()
        self.dummyDict.clear()
        self.lightManager.deleteAll()
        self.blendAnimDict.clear()
        self.particleDict.clear()
        self.particleNodes.clear()
        self.ModelNum = 0
        self.ActorNum = 0
        self.theScene = None
        messenger.send('SGE_Update Explorer', [render])
        print('Scene should be cleaned up!')

    def removeObj(self, nodePath):
        if False:
            return 10
        name = nodePath.getName()
        childrenList = nodePath.getChildren()
        if name in self.ModelDic:
            del self.ModelDic[name]
            del self.ModelRefDic[name]
            if len(childrenList) != 0:
                for node in childrenList:
                    self.removeObj(node)
            nodePath.removeNode()
            self.ModelNum -= 1
            pass
        elif name in self.ActorDic:
            del self.ActorDic[name]
            del self.ActorRefDic[name]
            if len(childrenList) != 0:
                for node in childrenList:
                    self.removeObj(node)
            nodePath.removeNode()
            self.ActorNum -= 1
            pass
        elif name in self.collisionDict:
            del self.collisionDict[name]
            if len(childrenList) != 0:
                for node in childrenList:
                    self.removeObj(node)
            nodePath.removeNode()
            pass
        elif name in self.dummyDict:
            del self.dummyDict[name]
            if len(childrenList) != 0:
                for node in childrenList:
                    self.removeObj(node)
            nodePath.removeNode()
            pass
        elif self.lightManager.isLight(name):
            if len(childrenList) != 0:
                for node in childrenList:
                    self.removeObj(node)
            list = self.lightManager.delete(name)
            return list
        elif name in self.particleNodes:
            self.particleNodes[name].removeNode()
            del self.particleNodes[name]
            del self.particleDict[name]
        else:
            print('You cannot remove this NodePath')
            return
        messenger.send('SGE_Update Explorer', [render])
        return

    def duplicateObj(self, nodePath, pos, hpr, scale, num):
        if False:
            while True:
                i = 10
        name = nodePath.getName()
        isModel = True
        cPos = pos
        cHpr = hpr
        cScale = scale
        parent = nodePath.getParent()
        if name in self.ActorDic:
            holder = self.ActorDic
            holderRef = self.ActorRefDic
            isModel = False
        elif name in self.ModelDic:
            holder = self.ModelDic
            holderRef = self.ModelRefDic
        else:
            print('---- DataHolder: Target Obj is not a legal object could be duplicate!')
            return
        FilePath = holderRef[name]
        oPos = holder[name].getPos() + cPos
        oHpr = holder[name].getHpr() + cHpr
        for i in range(num):
            if isModel:
                newName = name + '_copy_%d' % i
                while self.isInScene(newName):
                    newName = newName + '_1'
                holder[newName] = loader.loadModel(FilePath.getFullpath())
                holderRef[newName] = FilePath
                self.ModelNum += 1
                holder[newName].reparentTo(parent)
                holder[newName].setPos(oPos)
                holder[newName].setHpr(oHpr)
                holder[newName].setScale(cScale)
                holder[newName].setName(newName)
                oPos = oPos + cPos
                oHpr = oHpr + cHpr
            else:
                "\n                Yeah, Yeah, Yeah, I know I should not reload the Actor but get it from modelpool too.\n                I tried, but it caused some error.\n                I 'might' be back to fix this problem.\n                "
                newName = name + '_copy_%d' % i
                while self.isInScene(newName):
                    newName = newName + '_1'
                holder[newName] = Actor.Actor(FilePath.getFullpath())
                holderRef[newName] = FilePath
                self.ActorNum += 1
                holder[newName].reparentTo(parent)
                holder[newName].setPos(oPos)
                holder[newName].setHpr(oHpr)
                holder[newName].setScale(cScale)
                holder[newName].setName(newName)
                oPos = oPos + cPos
                oHpr = oHpr + cHpr
        messenger.send('SGE_Update Explorer', [render])
        return

    def loadModel(self, lFilePath, FilePath, Name='Model_'):
        if False:
            print('Hello World!')
        self.ModelNum += 1
        defaultName = Name + '%d' % self.ModelNum
        while self.isInScene(defaultName):
            defaultName = defaultName + '_1'
        self.ModelDic[defaultName] = loader.loadModel(FilePath)
        if self.ModelDic[defaultName] == None:
            del self.ModelDic[defaultName]
            self.ModelNum -= 1
            return False
        self.ModelRefDic[defaultName] = FilePath
        self.ModelDic[defaultName].setName(defaultName)
        self.ModelDic[defaultName].reparentTo(render)
        messenger.send('SGE_Update Explorer', [render])
        messenger.send('DH_LoadingComplete', [self.ModelDic[defaultName]])
        return True

    def loadActor(self, lFilePath, FilePath, Name='Actor_'):
        if False:
            while True:
                i = 10
        self.ActorNum += 1
        defaultName = Name + '%d' % self.ActorNum
        while self.isInScene(defaultName):
            defaultName = defaultName + '_1'
        self.ActorDic[defaultName] = Actor.Actor(FilePath.getFullpath())
        if self.ActorDic[defaultName] == None:
            del self.ActorDic[defaultName]
            self.ActorNum -= 1
            return False
        self.ActorRefDic[defaultName] = FilePath
        self.ActorDic[defaultName].setName(defaultName)
        self.ActorDic[defaultName].reparentTo(render)
        messenger.send('SGE_Update Explorer', [render])
        messenger.send('DH_LoadingComplete', [self.ActorDic[defaultName]])
        return True

    def isActor(self, name):
        if False:
            while True:
                i = 10
        return name in self.ActorDic

    def getActor(self, name):
        if False:
            return 10
        if self.isActor(name):
            return self.ActorDic[name]
        else:
            print('----No Actor named: ', name)
            return None

    def getModel(self, name):
        if False:
            while True:
                i = 10
        if self.isModel(name):
            return self.ModelDic[name]
        else:
            print('----No Model named: ', name)
            return None

    def isModel(self, name):
        if False:
            for i in range(10):
                print('nop')
        return name in self.ModelDic

    def loadAnimation(self, name, Dic):
        if False:
            for i in range(10):
                print('nop')
        if self.isActor(name):
            self.ActorDic[name].loadAnims(Dic)
            for anim in Dic:
                self.ActorDic[name].bindAnim(anim)
            messenger.send('DataH_loadFinish' + name)
            return
        else:
            print('------ Error when loading animation for Actor: ', name)

    def removeAnimation(self, name, anim):
        if False:
            for i in range(10):
                print('nop')
        if self.isActor(name):
            self.ActorDic[name].unloadAnims([anim])
            AnimDict = self.ActorDic[name].getAnimControlDict()
            del AnimDict['lodRoot']['modelRoot'][anim]
            messenger.send('DataH_removeAnimFinish' + name)
            messenger.send('animRemovedFromNode', [self.ActorDic[name], self.getAnimationDictFromActor(name)])
        return

    def toggleLight(self):
        if False:
            while True:
                i = 10
        self.lightManager.toggle()
        return

    def isLight(self, name):
        if False:
            i = 10
            return i + 15
        return self.lightManager.isLight(name)

    def createLight(self, type='ambient', lightcolor=VBase4(0.3, 0.3, 0.3, 1), specularColor=VBase4(1), position=Point3(0, 0, 0), orientation=Vec3(1, 0, 0), constant=1.0, linear=0.0, quadratic=0.0, exponent=0.0):
        if False:
            print('Hello World!')
        (list, lightNode) = self.lightManager.create(type, lightcolor, specularColor, position, orientation, constant, linear, quadratic, exponent)
        messenger.send('SGE_Update Explorer', [render])
        return (list, lightNode)

    def getLightList(self):
        if False:
            i = 10
            return i + 15
        return self.lightManager.getLightList()

    def getLightNode(self, lightName):
        if False:
            return 10
        return self.lightManager.getLightNode(lightName)

    def toggleLightNode(self, lightNode):
        if False:
            while True:
                i = 10
        if lightNode.active:
            self.lightManager.setOff(lightNode)
        else:
            self.lightManager.setOn(lightNode)
        return

    def rename(self, nodePath, nName):
        if False:
            print('Hello World!')
        oName = nodePath.getName()
        if oName == nName:
            return
        while self.isInScene(nName):
            nName = nName + '_1'
        if self.isActor(oName):
            self.ActorDic[nName] = self.ActorDic[oName]
            self.ActorRefDic[nName] = self.ActorRefDic[oName]
            self.ActorDic[nName].setName(nName)
            if oName in self.blendAnimDict:
                self.blendAnimDict[nName] = self.blendAnimDict[oName]
                del self.blendAnimDict[oName]
            del self.ActorDic[oName]
            del self.ActorRefDic[oName]
        elif self.isModel(oName):
            self.ModelDic[nName] = self.ModelDic[oName]
            self.ModelRefDic[nName] = self.ModelRefDic[oName]
            self.ModelDic[nName].setName(nName)
            del self.ModelDic[oName]
            del self.ModelRefDic[oName]
        elif self.lightManager.isLight(oName):
            (list, lightNode) = self.lightManager.rename(oName, nName)
        elif oName in self.dummyDict:
            self.dummyDict[nName] = self.dummyDict[oName]
            self.dummyDict[nName].setName(nName)
            del self.dummyDict[oName]
        elif oName in self.collisionDict:
            self.collisionDict[nName] = self.collisionDict[oName]
            self.collisionDict[nName].setName(nName)
            del self.collisionDict[oName]
        elif oName in self.particleNodes:
            self.particleNodes[nName] = self.particleNodes[oName]
            self.particleDict[nName] = self.particleDict[oName]
            self.particleDict[nName].setName(nName)
            self.particleNodes[nName].setName(nName)
            del self.particleNodes[oName]
            del self.particleDict[oName]
        else:
            print('----Error: This Object is not allowed to this function!')
        if oName in self.curveDict:
            self.curveDict[nName] = self.curveDict[oName]
            del self.curveDict[oName]
        if self.lightManager.isLight(nName):
            return (list, lightNode)

    def isInScene(self, name):
        if False:
            print('Hello World!')
        if self.isActor(name):
            return True
        elif self.isModel(name):
            return True
        elif self.lightManager.isLight(name):
            return True
        elif name in self.dummyDict:
            return True
        elif name in self.collisionDict:
            return True
        elif name in self.particleNodes:
            return True
        elif name == 'render' or name == 'SEditor' or name == 'Lights' or (name == 'camera'):
            return True
        return False

    def bindCurveToNode(self, node, curveCollection):
        if False:
            while True:
                i = 10
        name = node.getName()
        if name in self.curveDict:
            self.curveDict[name].append(curveCollection)
            return
        else:
            self.curveDict[name] = [curveCollection]
            return
        return

    def getCurveList(self, nodePath):
        if False:
            i = 10
            return i + 15
        name = nodePath.getName()
        if name in self.curveDict:
            return self.curveDict[name]
        else:
            return None

    def removeCurveFromNode(self, nodePath, curveName):
        if False:
            i = 10
            return i + 15
        name = nodePath.getName()
        if name in self.curveDict:
            index = None
            for curve in self.curveDict[name]:
                if curve.getCurve(0).getName() == curveName:
                    index = self.curveDict[name].index(curve)
                    break
            del self.curveDict[name][index]
            if len(self.curveDict[name]) != 0:
                messenger.send('curveRemovedFromNode', [nodePath, self.curveDict[name]])
            else:
                del self.curveDict[name]
                messenger.send('curveRemovedFromNode', [nodePath, None])
        return

    def getInfoOfThisNode(self, nodePath):
        if False:
            print('Hello World!')
        type = ''
        info = {}
        name = nodePath.getName()
        if name == 'render':
            type = 'render'
        elif name == 'camera':
            type = 'camera'
            cameraNode = base.cam.node()
            lens = cameraNode.getLens()
            info['lensType'] = lens.getClassType().getName()
            info['far'] = lens.getFar()
            info['near'] = lens.getNear()
            info['FilmSize'] = lens.getFilmSize()
            info['fov'] = lens.getFov()
            info['hFov'] = lens.getHfov()
            info['vFov'] = lens.getVfov()
            info['focalLength'] = lens.getFocalLength()
        elif name == 'SEditor':
            type = 'Special'
        elif self.isActor(name):
            type = 'Actor'
            info['filePath'] = self.ActorRefDic[name]
            info['animDict'] = self.getAnimationDictFromActor(name)
        elif self.isModel(name):
            type = 'Model'
            info['filePath'] = self.ModelRefDic[name]
        elif self.isLight(name):
            type = 'Light'
            info['lightNode'] = self.lightManager.getLightNode(name)
        elif name in self.dummyDict:
            type = 'dummy'
        elif name in self.collisionDict:
            type = 'collisionNode'
            info['collisionNode'] = self.collisionDict[name]
        if name in self.curveDict:
            info['curveList'] = self.getCurveList(nodePath)
        return (type, info)

    def getAnimationDictFromActor(self, actorName):
        if False:
            while True:
                i = 10
        animContorlDict = self.ActorDic[actorName].getAnimControlDict()
        animNameList = self.ActorDic[actorName].getAnimNames()
        if len(animNameList) == 0:
            return {}
        animDict = {}
        for anim in animNameList:
            animDict[anim] = animContorlDict['lodRoot']['modelRoot'][anim][0]
        return animDict

    def addDummyNode(self, nodePath):
        if False:
            for i in range(10):
                print('nop')
        number = len(self.dummyDict)
        number += 1
        name = 'Dummy%d' % number
        self.dummyModel = loader.loadModel('models/misc/sphere')
        self.dummyModel.reparentTo(nodePath)
        while self.isInScene(name):
            name = name + '_1'
        self.dummyModel.setName(name)
        self.dummyDict[name] = self.dummyModel
        messenger.send('SGE_Update Explorer', [render])
        return

    def addCollisionObject(self, collisionObj, nodePath, pointA=None, pointB=None, pointC=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        if name == None:
            name = 'CollisionNode_%d' % len(self.collisionDict)
        while self.isInScene(name):
            name = name + '_1'
        node = CollisionNode(name)
        node.addSolid(collisionObj)
        self.collisionDict[name] = nodePath.attachNewNode(node)
        if pointA != None:
            self.collisionDict[name].setTag('A_X', '%f' % pointA.getX())
            self.collisionDict[name].setTag('A_Y', '%f' % pointA.getY())
            self.collisionDict[name].setTag('A_Z', '%f' % pointA.getZ())
            self.collisionDict[name].setTag('B_X', '%f' % pointB.getX())
            self.collisionDict[name].setTag('B_Y', '%f' % pointB.getY())
            self.collisionDict[name].setTag('B_Z', '%f' % pointB.getZ())
            self.collisionDict[name].setTag('C_X', '%f' % pointC.getX())
            self.collisionDict[name].setTag('C_Y', '%f' % pointC.getY())
            self.collisionDict[name].setTag('C_Z', '%f' % pointC.getZ())
        if self.collisionVisable:
            self.collisionDict[name].show()
        base.cTrav.addCollider(self.collisionDict[name], self.CollisionHandler)
        messenger.send('SGE_Update Explorer', [render])
        return

    def toggleCollisionVisable(self, visable):
        if False:
            while True:
                i = 10
        if visable == 1:
            self.collisionVisable = True
            for name in self.collisionDict:
                if self.collisionDict[name].isHidden():
                    self.collisionDict[name].show()
        else:
            self.collisionVisable = False
            for name in self.collisionDict:
                if not self.collisionDict[name].isHidden():
                    self.collisionDict[name].hide()

    def toggleParticleVisable(self, visable):
        if False:
            while True:
                i = 10
        if not visable:
            for name in self.particleNodes:
                self.particleNodes[name].setTransparency(True)
                self.particleNodes[name].setAlphaScale(0)
                self.particleNodes[name].setBin('fixed', 1)
        else:
            for name in self.particleNodes:
                self.particleNodes[name].setTransparency(False)
                self.particleNodes[name].setAlphaScale(1)
                self.particleNodes[name].setBin('default', 1)
        return

    def getBlendAnimAsDict(self, name):
        if False:
            print('Hello World!')
        if name in self.blendAnimDict:
            return self.blendAnimDict[name]
        else:
            return {}

    def saveBlendAnim(self, actorName, blendName, animNameA, animNameB, effect):
        if False:
            i = 10
            return i + 15
        if actorName in self.blendAnimDict:
            if blendName in self.blendAnimDict[actorName]:
                self.blendAnimDict[actorName][blendName][0] = animNameA
                self.blendAnimDict[actorName][blendName][1] = animNameB
                self.blendAnimDict[actorName][blendName][2] = effect
            else:
                self.blendAnimDict[actorName][blendName] = [animNameA, animNameB, effect]
        else:
            self.getActor(actorName).setTag('Blending', 'True')
            self.blendAnimDict[actorName] = {blendName: [animNameA, animNameB, effect]}
        return self.blendAnimDict[actorName]

    def renameBlendAnim(self, actorName, nName, oName, animNameA, animNameB, effect):
        if False:
            return 10
        self.removeBlendAnim(actorName, oName)
        print(self.blendAnimDict)
        return self.saveBlendAnim(actorName, nName, animNameA, animNameB, effect)

    def removeBlendAnim(self, actorName, blendName):
        if False:
            while True:
                i = 10
        if actorName in self.blendAnimDict:
            if blendName in self.blendAnimDict[actorName]:
                del self.blendAnimDict[actorName][blendName]
            if len(self.blendAnimDict[actorName]) == 0:
                del self.blendAnimDict[actorName]
                self.getActor(actorName).clearTag('Blending')
                return {}
            return self.blendAnimDict[actorName]
        else:
            return {}

    def getAllObjNameAsList(self):
        if False:
            while True:
                i = 10
        list = ['camera']
        list = list + self.ModelDic.keys() + self.ActorDic.keys() + self.collisionDict.keys() + self.dummyDict.keys() + self.particleNodes.keys() + self.lightManager.getLightList()
        return list

    def getObjFromSceneByName(self, name):
        if False:
            return 10
        if name == 'camera':
            return camera
        elif name in self.ModelDic:
            return self.ModelDic[name]
        elif name in self.ActorDic:
            return self.ActorDic[name]
        elif name in self.collisionDict:
            return self.collisionDict[name]
        elif name in self.dummyDict:
            return self.dummyDict[name]
        elif name in self.particleNodes:
            return self.particleNodes[name]
        elif self.lightManager.isLight(name):
            return self.lightManager.getLightNode(name)
        return None

    def getControlSetting(self):
        if False:
            return 10
        if self.controlType == 'Keyboard':
            return (self.controlType, [self.controlTarget, self.keyboardMapDict.copy(), self.keyboardSpeedDict.copy()])
        elif self.controlType == 'Tracker':
            return (self.controlType, [])
        return

    def saveControlSetting(self, controlType, data):
        if False:
            while True:
                i = 10
        if controlType == 'Keyboard':
            self.controlType = controlType
            self.controlTarget = data[0]
            self.keyboardMapDict.clear()
            self.keyboardMapDict = data[1].copy()
            self.keyboardSpeedDict.clear()
            self.keyboardSpeedDict = data[2].copy()
            return

    def loadScene(self):
        if False:
            i = 10
            return i + 15
        OpenFilename = askopenfilename(filetypes=[('PY', 'py')], title='Load Scene')
        if not OpenFilename:
            return None
        f = Filename.fromOsSpecific(OpenFilename)
        fileName = f.getBasenameWoExtension()
        dirName = f.getFullpathWoExtension()
        print('DATAHOLDER::' + dirName)
        sys.path.append(os.path.dirname(f.toOsSpecific()))
        self.theScene = __import__(fileName)
        self.Scene = self.theScene.SavedScene(0, seParticleEffect, seParticles, dirName)
        messenger.send('SGE_Update Explorer', [render])
        for model in self.Scene.ModelDic:
            self.ModelDic[model] = self.Scene.ModelDic[model]
            self.ModelRefDic[model] = Filename(dirName + '/' + self.Scene.ModelRefDic[model])
            self.ModelNum = self.ModelNum + 1
        for actor in self.Scene.ActorDic:
            self.ActorDic[actor] = self.Scene.ActorDic[actor]
            self.ActorRefDic[actor] = Filename(dirName + '/' + self.Scene.ActorRefDic[actor])
            if str(actor) in self.Scene.blendAnimDict:
                self.blendAnimDict[actor] = self.Scene.blendAnimDict[actor]
            self.ActorNum = self.ActorNum + 1
        for light in self.Scene.LightDict:
            alight = self.Scene.LightDict[light]
            type = self.Scene.LightTypes[light]
            thenode = self.Scene.LightNodes[light]
            if type == 'ambient':
                self.lightManager.create('ambient', alight.getColor(), name=alight.getName(), tag=thenode.getTag('Metadata'))
            elif type == 'directional':
                self.lightManager.create('directional', alight.getColor(), alight.getSpecularColor(), thenode.getPos(), thenode.getHpr(), name=alight.getName(), tag=thenode.getTag('Metadata'))
            elif type == 'point':
                atten = alight.getAttenuation()
                self.lightManager.create('point', alight.getColor(), alight.getSpecularColor(), thenode.getPos(), Vec3(1, 0, 0), atten.getX(), atten.getY(), atten.getZ(), name=alight.getName(), tag=thenode.getTag('Metadata'))
            elif type == 'spot':
                atten = alight.getAttenuation()
                self.lightManager.create('spot', alight.getColor(), alight.getSpecularColor(), thenode.getPos(), thenode.getHpr(), atten.getX(), atten.getY(), atten.getZ(), alight.getExponent(), name=alight.getName(), tag=thenode.getTag('Metadata'))
            else:
                print('Invalid light type')
        for dummy in self.Scene.dummyDict:
            self.dummyDict[dummy] = self.Scene.dummyDict[dummy]
        for collnode in self.Scene.collisionDict:
            self.collisionDict[collnode] = self.Scene.collisionDict[collnode]
        for node in self.Scene.curveDict:
            curveCollection = self.Scene.curveDict[node]
            for curve in curveCollection:
                curveColl = ParametricCurveCollection()
                nodeP = loader.loadModel(curve)
                curveColl.addCurves(nodeP.node())
                nodeP.removeNode()
                thenode = render.find('**/' + str(node))
                self.bindCurveToNode(thenode, curveColl)
        for effect in self.Scene.particleDict:
            theeffect = self.Scene.particleDict[effect]
            emitter = loader.loadModel('sphere')
            emitter.setPosHprScale(theeffect.getX(), theeffect.getY(), theeffect.getZ(), theeffect.getH(), theeffect.getP(), theeffect.getR(), theeffect.getSx(), theeffect.getSy(), theeffect.getSz())
            theeffect.setPos(0, 0, 0)
            theeffect.setName(str(effect))
            tempparent = theeffect.getParent()
            theeffect.reparentTo(emitter)
            emitter.setName(str(effect))
            emitter.reparentTo(tempparent)
            theeffect.enable()
            self.particleDict[effect] = theeffect
            self.particleNodes[effect] = emitter
        for light in self.Scene.LightDict:
            vestige = render.find('**/' + light)
            if vestige != None:
                vestige.removeNode()
        messenger.send('SGE_Update Explorer', [render])
        if OpenFilename:
            return OpenFilename
        else:
            return None

    def getList(self):
        if False:
            print('Hello World!')
        return self.lightManager.getList()