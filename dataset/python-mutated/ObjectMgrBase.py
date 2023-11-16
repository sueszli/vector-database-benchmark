"""
Defines ObjectMgrBase
"""
import os
import time
import copy
from panda3d.core import ConfigVariableString, Filename, Mat4, NodePath
from direct.actor.Actor import Actor
from direct.showbase.PythonUtil import Functor
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from .ActionMgr import ActionTransformObj, ActionUpdateObjectProp
from . import ObjectGlobals as OG

class PythonNodePath(NodePath):

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        NodePath.__init__(self, node)

class ObjectMgrBase:
    """ ObjectMgr will create, manage, update objects in the scene """

    def __init__(self, editor):
        if False:
            for i in range(10):
                print('nop')
        self.editor = editor
        self.objects = {}
        self.npIndex = {}
        self.saveData = []
        self.objectsLastXform = {}
        self.lastUid = ''
        self.lastUidMode = 0
        self.currNodePath = None
        self.currLiveNP = None
        self.Actor = []
        self.findActors(base.render)
        self.Nodes = []
        self.findNodes(base.render)

    def reset(self):
        if False:
            return 10
        base.direct.deselectAllCB()
        for id in list(self.objects.keys()):
            try:
                self.objects[id][OG.OBJ_NP].removeNode()
            except Exception:
                pass
            del self.objects[id]
        for np in list(self.npIndex.keys()):
            del self.npIndex[np]
        self.objects = {}
        self.npIndex = {}
        self.saveData = []
        self.Actor = []
        self.Nodes = []

    def genUniqueId(self):
        if False:
            while True:
                i = 10
        userId = os.path.basename(os.path.expandvars('$USERNAME'))
        if userId == '':
            userId = ConfigVariableString('le-user-id').value
        if userId == '':
            userId = 'unknown'
        newUid = str(time.time()) + userId
        if self.lastUid == newUid:
            newUid = newUid + str(self.lastUidMod)
            self.lastUidMod = self.lastUidMod + 1
        else:
            self.lastUid = newUid
            self.lastUidMod = 0
        return newUid

    def addNewCurveFromFile(self, curveInfo, degree, uid=None, parent=None, fSelectObject=True, nodePath=None):
        if False:
            i = 10
            return i + 15
        ' function to add new curve to the scene from file'
        curve = []
        curveControl = []
        for item in curveInfo:
            controler = base.render.attachNewNode('controler')
            controler = base.loader.loadModel('models/misc/smiley')
            controlerPathname = f'controler{item[0]}'
            controler.setName(controlerPathname)
            controler.setPos(item[1])
            controler.setColor(0, 0, 0, 1)
            controler.setScale(0.2)
            controler.reparentTo(base.render)
            controler.setTag('OBJRoot', '1')
            controler.setTag('Controller', '1')
            curve.append((None, item[1]))
            curveControl.append((item[0], controler))
        self.editor.curveEditor.degree = degree
        self.editor.curveEditor.ropeUpdate(curve)
        curveObjNP = self.addNewCurve(curveControl, degree, uid, parent, fSelectObject, nodePath=self.editor.curveEditor.currentRope)
        curveObj = self.findObjectByNodePath(curveObjNP)
        self.editor.objectMgr.updateObjectPropValue(curveObj, 'Degree', degree, fSelectObject=False, fUndo=False)
        for item in curveControl:
            item[1].reparentTo(curveObjNP)
            item[1].hide()
        curveControl = []
        curve = []
        self.editor.curveEditor.currentRope = None
        return curveObjNP

    def addNewCurve(self, curveInfo, degree, uid=None, parent=None, fSelectObject=True, nodePath=None):
        if False:
            print('Hello World!')
        ' function to add new curve to the scene'
        if parent is None:
            parent = self.editor.NPParent
        if uid is None:
            uid = self.genUniqueId()
        if self.editor:
            objDef = self.editor.objectPalette.findItem('__Curve__')
        if nodePath is None:
            pass
        else:
            newobj = nodePath
        newobj.reparentTo(parent)
        newobj.setTag('OBJRoot', '1')
        properties = {}
        for key in objDef.properties.keys():
            properties[key] = objDef.properties[key][OG.PROP_DEFAULT]
        properties['Degree'] = degree
        properties['curveInfo'] = curveInfo
        self.objects[uid] = [uid, newobj, objDef, None, None, properties, (1, 1, 1, 1)]
        self.npIndex[NodePath(newobj)] = uid
        if self.editor:
            if fSelectObject:
                self.editor.select(newobj, fUndo=0)
            self.editor.ui.sceneGraphUI.add(newobj, parent)
            self.editor.fNeedToSave = True
        return newobj

    def addNewObject(self, typeName, uid=None, model=None, parent=None, anim=None, fSelectObject=True, nodePath=None, nameStr=None):
        if False:
            for i in range(10):
                print('nop')
        ' function to add new obj to the scene '
        if parent is None:
            parent = self.editor.NPParent
        if uid is None:
            uid = self.genUniqueId()
        if self.editor:
            objDef = self.editor.objectPalette.findItem(typeName)
            if objDef is None:
                objDef = self.editor.protoPalette.findItem(typeName)
        else:
            objDef = base.objectPalette.findItem(typeName)
            if objDef is None:
                objDef = base.protoPalette.findItem(typeName)
        newobj = None
        if objDef and (not isinstance(objDef, dict)):
            if not hasattr(objDef, 'createFunction'):
                return newobj
            if nodePath is None:
                if objDef.createFunction:
                    funcName = objDef.createFunction[OG.FUNC_NAME]
                    funcArgs = copy.deepcopy(objDef.createFunction[OG.FUNC_ARGS])
                    for pair in list(funcArgs.items()):
                        if pair[1] == OG.ARG_NAME:
                            funcArgs[pair[0]] = nameStr
                        elif pair[1] == OG.ARG_PARENT:
                            funcArgs[pair[0]] = parent
                    if isinstance(funcName, str):
                        if funcName.startswith('.'):
                            if self.editor:
                                func = Functor(getattr(self.editor, 'objectHandler%s' % funcName))
                            else:
                                func = Functor(getattr(base, 'objectHandler%s' % funcName))
                        else:
                            func = Functor(eval(funcName))
                    else:
                        func = funcName
                    newobj = func(**funcArgs)
                elif objDef.actor:
                    if model is None:
                        model = objDef.model
                    try:
                        newobj = Actor(model)
                    except Exception:
                        newobj = Actor(Filename.fromOsSpecific(model).getFullpath())
                    if hasattr(objDef, 'animDict') and objDef.animDict != {}:
                        objDef.anims = objDef.animDict.get(model)
                elif objDef.model is not None:
                    if model is None:
                        model = objDef.model
                    try:
                        newobjModel = base.loader.loadModel(model)
                    except Exception:
                        newobjModel = base.loader.loadModel(Filename.fromOsSpecific(model).getFullpath(), okMissing=True)
                    if newobjModel:
                        self.flatten(newobjModel, model, objDef, uid)
                        newobj = PythonNodePath(newobjModel)
                    else:
                        newobj = None
                else:
                    newobj = hidden.attachNewNode(objDef.name)
            else:
                newobj = nodePath
            i = 0
            for i in range(len(objDef.anims)):
                animFile = objDef.anims[i]
                animName = os.path.basename(animFile)
                if i < len(objDef.animNames):
                    animName = objDef.animNames[i]
                newAnim = newobj.loadAnims({animName: animFile})
                if anim:
                    if anim == animFile:
                        newobj.loop(animName)
                elif i == 0:
                    anim = animFile
                    newobj.loop(animName)
            if newobj is None:
                return None
            newobj.reparentTo(parent)
            newobj.setTag('OBJRoot', '1')
            properties = {}
            for key in objDef.properties.keys():
                properties[key] = objDef.properties[key][OG.PROP_DEFAULT]
            self.objects[uid] = [uid, newobj, objDef, model, anim, properties, (1, 1, 1, 1)]
            self.npIndex[NodePath(newobj)] = uid
            if self.editor:
                if fSelectObject:
                    self.editor.select(newobj, fUndo=0)
                self.editor.ui.sceneGraphUI.add(newobj, parent)
                self.editor.fNeedToSave = True
        return newobj

    def removeObjectById(self, uid):
        if False:
            return 10
        obj = self.findObjectById(uid)
        nodePath = obj[OG.OBJ_NP]
        for i in range(0, len(self.Actor)):
            if self.Actor[i] == obj:
                del self.Actor[i]
                break
        for i in range(0, len(self.Nodes)):
            if self.Nodes[i][OG.OBJ_UID] == uid:
                del self.Nodes[i]
                break
        self.editor.animMgr.removeAnimInfo(obj[OG.OBJ_UID])
        del self.objects[uid]
        del self.npIndex[nodePath]
        for child in nodePath.getChildren():
            if child.hasTag('OBJRoot'):
                self.removeObjectByNodePath(child)
        nodePath.remove()
        self.editor.fNeedToSave = True

    def removeObjectByNodePath(self, nodePath):
        if False:
            while True:
                i = 10
        uid = self.npIndex.get(nodePath)
        if uid:
            for i in range(0, len(self.Actor)):
                if self.Actor[i][OG.OBJ_UID] == uid:
                    del self.Actor[i]
                    break
            for i in range(0, len(self.Nodes)):
                if self.Nodes[i][OG.OBJ_UID] == uid:
                    del self.Nodes[i]
                    break
            self.editor.animMgr.removeAnimInfo(uid)
            del self.objects[uid]
            del self.npIndex[nodePath]
        for child in nodePath.getChildren():
            if child.hasTag('OBJRoot'):
                self.removeObjectByNodePath(child)
        self.editor.fNeedToSave = True

    def findObjectById(self, uid):
        if False:
            print('Hello World!')
        return self.objects.get(uid)

    def findObjectByNodePath(self, nodePath):
        if False:
            return 10
        uid = self.npIndex.get(NodePath(nodePath))
        if uid is None:
            return None
        else:
            return self.objects[uid]

    def findObjectByNodePathBelow(self, nodePath):
        if False:
            for i in range(10):
                print('nop')
        for ancestor in nodePath.getAncestors():
            if ancestor.hasTag('OBJRoot'):
                return self.findObjectByNodePath(ancestor)
        return None

    def findObjectsByTypeName(self, typeName):
        if False:
            while True:
                i = 10
        results = []
        for uid in self.objects.keys():
            obj = self.objects[uid]
            if obj[OG.OBJ_DEF].name == typeName:
                results.append(obj)
        return results

    def deselectAll(self):
        if False:
            i = 10
            return i + 15
        self.currNodePath = None
        taskMgr.remove('_le_updateObjectUITask')
        self.editor.ui.objectPropertyUI.clearPropUI()
        self.editor.ui.sceneGraphUI.tree.UnselectAll()

    def selectObject(self, nodePath, fLEPane=0):
        if False:
            print('Hello World!')
        obj = self.findObjectByNodePath(nodePath)
        if obj is None:
            return
        self.selectObjectCB(obj, fLEPane)

    def selectObjectCB(self, obj, fLEPane):
        if False:
            while True:
                i = 10
        self.currNodePath = obj[OG.OBJ_NP]
        self.objectsLastXform[obj[OG.OBJ_UID]] = Mat4(self.currNodePath.getMat())
        self.spawnUpdateObjectUITask()
        self.updateObjectPropertyUI(obj)
        if fLEPane == 0:
            self.editor.ui.sceneGraphUI.select(obj[OG.OBJ_UID])
        if not obj[OG.OBJ_DEF].movable:
            if base.direct.widget.fActive:
                base.direct.widget.toggleWidget()

    def updateObjectPropertyUI(self, obj):
        if False:
            return 10
        objDef = obj[OG.OBJ_DEF]
        objProp = obj[OG.OBJ_PROP]
        self.editor.ui.objectPropertyUI.updateProps(obj, objDef.movable)
        self.editor.fNeedToSave = True

    def onEnterObjectPropUI(self, event):
        if False:
            while True:
                i = 10
        taskMgr.remove('_le_updateObjectUITask')
        self.editor.ui.bindKeyEvents(False)

    def onLeaveObjectPropUI(self, event):
        if False:
            return 10
        self.spawnUpdateObjectUITask()
        self.editor.ui.bindKeyEvents(True)

    def spawnUpdateObjectUITask(self):
        if False:
            return 10
        if self.currNodePath is None:
            return
        taskMgr.remove('_le_updateObjectUITask')
        t = Task.Task(self.updateObjectUITask)
        t.np = self.currNodePath
        taskMgr.add(t, '_le_updateObjectUITask')

    def updateObjectUITask(self, state):
        if False:
            print('Hello World!')
        self.editor.ui.objectPropertyUI.propX.setValue(state.np.getX())
        self.editor.ui.objectPropertyUI.propY.setValue(state.np.getY())
        self.editor.ui.objectPropertyUI.propZ.setValue(state.np.getZ())
        h = state.np.getH()
        while h < 0:
            h = h + 360.0
        while h > 360:
            h = h - 360.0
        p = state.np.getP()
        while p < 0:
            p = p + 360.0
        while p > 360:
            p = p - 360.0
        r = state.np.getR()
        while r < 0:
            r = r + 360.0
        while r > 360:
            r = r - 360.0
        self.editor.ui.objectPropertyUI.propH.setValue(h)
        self.editor.ui.objectPropertyUI.propP.setValue(p)
        self.editor.ui.objectPropertyUI.propR.setValue(r)
        self.editor.ui.objectPropertyUI.propSX.setValue(state.np.getSx())
        self.editor.ui.objectPropertyUI.propSY.setValue(state.np.getSy())
        self.editor.ui.objectPropertyUI.propSZ.setValue(state.np.getSz())
        return Task.cont

    def updateObjectTransform(self, event):
        if False:
            print('Hello World!')
        if self.currNodePath is None:
            return
        np = hidden.attachNewNode('temp')
        np.setX(float(self.editor.ui.objectPropertyUI.propX.getValue()))
        np.setY(float(self.editor.ui.objectPropertyUI.propY.getValue()))
        np.setZ(float(self.editor.ui.objectPropertyUI.propZ.getValue()))
        h = float(self.editor.ui.objectPropertyUI.propH.getValue())
        while h < 0:
            h = h + 360.0
        while h > 360:
            h = h - 360.0
        p = float(self.editor.ui.objectPropertyUI.propP.getValue())
        while p < 0:
            p = p + 360.0
        while p > 360:
            p = p - 360.0
        r = float(self.editor.ui.objectPropertyUI.propR.getValue())
        while r < 0:
            r = r + 360.0
        while r > 360:
            r = r - 360.0
        np.setH(h)
        np.setP(p)
        np.setR(r)
        np.setSx(float(self.editor.ui.objectPropertyUI.propSX.getValue()))
        np.setSy(float(self.editor.ui.objectPropertyUI.propSY.getValue()))
        np.setSz(float(self.editor.ui.objectPropertyUI.propSZ.getValue()))
        obj = self.findObjectByNodePath(self.currNodePath)
        action = ActionTransformObj(self.editor, obj[OG.OBJ_UID], Mat4(np.getMat()))
        self.editor.actionMgr.push(action)
        np.remove()
        action()
        self.editor.fNeedToSave = True

    def setObjectTransform(self, uid, xformMat):
        if False:
            print('Hello World!')
        obj = self.findObjectById(uid)
        if obj:
            obj[OG.OBJ_NP].setMat(xformMat)
        self.editor.fNeedToSave = True

    def updateObjectColor(self, r, g, b, a, np=None):
        if False:
            print('Hello World!')
        if np is None:
            np = self.currNodePath
        obj = self.findObjectByNodePath(np)
        if not obj:
            return
        obj[OG.OBJ_RGBA] = (r, g, b, a)
        for child in np.getChildren():
            if not child.hasTag('OBJRoot') and (not child.hasTag('_le_sys')) and (child.getName() != 'bboxLines'):
                child.setTransparency(1)
                child.setColorScale(r, g, b, a)
        self.editor.fNeedToSave = True

    def updateObjectModel(self, model, obj, fSelectObject=True):
        if False:
            return 10
        " replace object's model "
        if obj[OG.OBJ_MODEL] != model:
            base.direct.deselectAllCB()
            objNP = obj[OG.OBJ_NP]
            objDef = obj[OG.OBJ_DEF]
            objRGBA = obj[OG.OBJ_RGBA]
            uid = obj[OG.OBJ_UID]
            if objDef.actor:
                try:
                    newobj = Actor(model)
                except Exception:
                    newobj = Actor(Filename.fromOsSpecific(model).getFullpath())
            else:
                newobjModel = base.loader.loadModel(model, okMissing=True)
                if newobjModel is None:
                    print("Can't load model %s" % model)
                    return
                self.flatten(newobjModel, model, objDef, uid)
                newobj = PythonNodePath(newobjModel)
            newobj.setTag('OBJRoot', '1')
            objNP.findAllMatches('=OBJRoot').reparentTo(newobj)
            newobj.reparentTo(objNP.getParent())
            newobj.setPos(objNP.getPos())
            newobj.setHpr(objNP.getHpr())
            newobj.setScale(objNP.getScale())
            self.updateObjectColor(objRGBA[0], objRGBA[1], objRGBA[2], objRGBA[3], newobj)
            del self.npIndex[NodePath(objNP)]
            objNP.removeNode()
            obj[OG.OBJ_NP] = newobj
            obj[OG.OBJ_MODEL] = model
            self.npIndex[NodePath(newobj)] = obj[OG.OBJ_UID]
            self.editor.ui.sceneGraphUI.changeLabel(obj[OG.OBJ_UID], newobj.getName())
            self.editor.fNeedToSave = True
            animList = obj[OG.OBJ_DEF].animDict.get(model)
            if animList:
                self.updateObjectAnim(animList[0], obj, fSelectObject=fSelectObject)
            elif fSelectObject:
                base.direct.select(newobj, fUndo=0)

    def updateObjectAnim(self, anim, obj, fSelectObject=True):
        if False:
            return 10
        " replace object's anim "
        if obj[OG.OBJ_ANIM] != anim:
            base.direct.deselectAllCB()
            objNP = obj[OG.OBJ_NP]
            animName = os.path.basename(anim)
            newAnim = objNP.loadAnims({animName: anim})
            objNP.loop(animName)
            obj[OG.OBJ_ANIM] = anim
            if fSelectObject:
                base.direct.select(objNP, fUndo=0)
            self.editor.fNeedToSave = True

    def updateObjectModelFromUI(self, event, obj):
        if False:
            return 10
        " replace object's model with one selected from UI "
        model = event.GetString()
        if model is not None:
            self.updateObjectModel(model, obj)

    def updateObjectAnimFromUI(self, event, obj):
        if False:
            print('Hello World!')
        " replace object's anim with one selected from UI "
        anim = event.GetString()
        if anim is not None:
            self.updateObjectAnim(anim, obj)

    def updateObjectProperty(self, event, obj, propName):
        if False:
            for i in range(10):
                print('nop')
        "\n        When an obj's property is updated in UI,\n        this will update it's value in data structure.\n        And call update function if defined.\n        "
        objDef = obj[OG.OBJ_DEF]
        objProp = obj[OG.OBJ_PROP]
        propDef = objDef.properties[propName]
        if propDef is None:
            return
        propType = propDef[OG.PROP_TYPE]
        propDataType = propDef[OG.PROP_DATATYPE]
        if propType == OG.PROP_UI_SLIDE:
            if len(propDef) <= OG.PROP_RANGE:
                return
            strVal = event.GetString()
            if strVal == '':
                min = float(propDef[OG.PROP_RANGE][OG.RANGE_MIN])
                max = float(propDef[OG.PROP_RANGE][OG.RANGE_MAX])
                intVal = event.GetInt()
                if intVal is None:
                    return
                val = intVal / 100.0 * (max - min) + min
            else:
                val = strVal
        elif propType == OG.PROP_UI_ENTRY:
            val = event.GetString()
        elif propType == OG.PROP_UI_SPIN:
            val = event.GetInt()
        elif propType == OG.PROP_UI_CHECK:
            if event.GetInt():
                val = True
            else:
                val = False
        elif propType == OG.PROP_UI_RADIO:
            val = event.GetString()
        elif propType == OG.PROP_UI_COMBO:
            val = event.GetString()
        elif propType == OG.PROP_UI_COMBO_DYNAMIC:
            val = event.GetString()
        else:
            return
        self.updateObjectPropValue(obj, propName, val, fSelectObject=propType != OG.PROP_UI_SLIDE)

    def updateObjectPropValue(self, obj, propName, val, fSelectObject=False, fUndo=True):
        if False:
            i = 10
            return i + 15
        '\n        Update object property value and\n        call update function if defined.\n        '
        objDef = obj[OG.OBJ_DEF]
        objProp = obj[OG.OBJ_PROP]
        propDef = objDef.properties[propName]
        propDataType = propDef[OG.PROP_DATATYPE]
        if propDataType != OG.PROP_BLIND:
            val = OG.TYPE_CONV[propDataType](val)
            oldVal = objProp[propName]
            if propDef[OG.PROP_FUNC] is None:
                func = None
                undoFunc = None
            else:
                funcName = propDef[OG.PROP_FUNC][OG.FUNC_NAME]
                funcArgs = propDef[OG.PROP_FUNC][OG.FUNC_ARGS]
                kwargs = {}
                undoKwargs = {}
                for key in funcArgs.keys():
                    if funcArgs[key] == OG.ARG_VAL:
                        kwargs[key] = val
                        undoKwargs[key] = oldVal
                    elif funcArgs[key] == OG.ARG_OBJ:
                        undoKwargs[key] = obj
                        objProp[propName] = val
                        kwargs[key] = obj
                    elif funcArgs[key] == OG.ARG_NOLOADING:
                        kwargs[key] = fSelectObject
                        undoKwargs[key] = fSelectObject
                    else:
                        kwargs[key] = funcArgs[key]
                        undoKwargs[key] = funcArgs[key]
                if isinstance(funcName, str):
                    if funcName.startswith('.'):
                        if self.editor:
                            func = Functor(getattr(self.editor, 'objectHandler%s' % funcName), **kwargs)
                            undoFunc = Functor(getattr(self.editor, 'objectHandler%s' % funcName), **undoKwargs)
                        else:
                            func = Functor(getattr(base, 'objectHandler%s' % funcName), **kwargs)
                            undoFunc = Functor(getattr(base, '.objectHandler%s' % funcName), **undoKwargs)
                    else:
                        func = Functor(eval(funcName), **kwargs)
                        undoFunc = Functor(eval(funcName), **undoKwargs)
                else:
                    func = Functor(funcName, **kwargs)
                    undoFunc = Functor(funcName, **undoKwargs)
        else:
            oldVal = objProp[propName]
            func = None
            undoFunc = None
        action = ActionUpdateObjectProp(self.editor, fSelectObject, obj, propName, val, oldVal, func, undoFunc)
        if fUndo:
            self.editor.actionMgr.push(action)
        action()
        if self.editor:
            self.editor.fNeedToSave = True
            if fSelectObject:
                base.direct.select(obj[OG.OBJ_NP], fUndo=0)

    def updateCurve(self, val, obj):
        if False:
            for i in range(10):
                print('nop')
        curve = obj[OG.OBJ_NP]
        degree = int(val)
        curveNode = obj[OG.OBJ_PROP]['curveInfo']
        curveInfor = []
        for item in curveNode:
            curveInfor.append((None, item[1].getPos()))
        curve.setup(degree, curveInfor)

    def updateObjectProperties(self, nodePath, propValues):
        if False:
            i = 10
            return i + 15
        "\n        When a saved level is loaded,\n        update an object's properties\n        And call update function if defined.\n        "
        obj = self.findObjectByNodePath(nodePath)
        if obj:
            for propName in propValues:
                self.updateObjectPropValue(obj, propName, propValues[propName])

    def traverse(self, parent, parentId=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Trasverse scene graph to gather data for saving\n        '
        for child in parent.getChildren():
            if child.hasTag('OBJRoot') and (not child.hasTag('Controller')):
                obj = self.findObjectByNodePath(child)
                if obj:
                    uid = obj[OG.OBJ_UID]
                    np = obj[OG.OBJ_NP]
                    objDef = obj[OG.OBJ_DEF]
                    objModel = obj[OG.OBJ_MODEL]
                    objAnim = obj[OG.OBJ_ANIM]
                    objProp = obj[OG.OBJ_PROP]
                    objRGBA = obj[OG.OBJ_RGBA]
                    if parentId:
                        parentStr = "objects['%s']" % parentId
                    else:
                        parentStr = 'None'
                    if objModel:
                        modelStr = "'%s'" % objModel
                    else:
                        modelStr = 'None'
                    if objAnim:
                        animStr = "'%s'" % objAnim
                    else:
                        animStr = 'None'
                    if objDef.named:
                        nameStr = "'%s'" % np.getName()
                    else:
                        nameStr = 'None'
                    if objDef.name == '__Curve__':
                        objCurveInfo = obj[OG.OBJ_PROP]['curveInfo']
                        self.objDegree = obj[OG.OBJ_PROP]['Degree']
                        newobjCurveInfo = []
                        for item in objCurveInfo:
                            newobjCurveInfo.append((item[0], item[1].getPos()))
                        self.saveData.append("\nobjects['%s'] = objectMgr.addNewCurveFromFile(%s, %s, '%s', %s, False, None)" % (uid, newobjCurveInfo, self.objDegree, uid, parentStr))
                    else:
                        self.saveData.append("\nobjects['%s'] = objectMgr.addNewObject('%s', '%s', %s, %s, %s, False, None, %s)" % (uid, objDef.name, uid, modelStr, parentStr, animStr, nameStr))
                    self.saveData.append("if objects['%s']:" % uid)
                    self.saveData.append("    objects['%s'].setPos(%s)" % (uid, np.getPos()))
                    self.saveData.append("    objects['%s'].setHpr(%s)" % (uid, np.getHpr()))
                    self.saveData.append("    objects['%s'].setScale(%s)" % (uid, np.getScale()))
                    self.saveData.append("    objectMgr.updateObjectColor(%f, %f, %f, %f, objects['%s'])" % (objRGBA[0], objRGBA[1], objRGBA[2], objRGBA[3], uid))
                    if objDef.name == '__Curve__':
                        pass
                    else:
                        self.saveData.append("    objectMgr.updateObjectProperties(objects['%s'], %s)" % (uid, objProp))
                self.traverse(child, uid)

    def getSaveData(self):
        if False:
            i = 10
            return i + 15
        self.saveData = []
        self.getPreSaveData()
        self.traverse(base.render)
        self.getPostSaveData()
        return self.saveData

    def getPreSaveData(self):
        if False:
            i = 10
            return i + 15
        '\n        if there are additional data to be saved before main data\n        you can override this function to populate data\n        '

    def getPostSaveData(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        if there are additional data to be saved after main data\n        you can override this function to populate data\n        '

    def duplicateObject(self, nodePath, parent=None):
        if False:
            for i in range(10):
                print('nop')
        obj = self.findObjectByNodePath(nodePath)
        if obj is None:
            return None
        objDef = obj[OG.OBJ_DEF]
        objModel = obj[OG.OBJ_MODEL]
        objAnim = obj[OG.OBJ_ANIM]
        objRGBA = obj[OG.OBJ_RGBA]
        if parent is None:
            parentNP = nodePath.getParent()
            parentObj = self.findObjectByNodePath(parentNP)
            if parentObj is None:
                parent = parentNP
            else:
                parent = parentObj[OG.OBJ_NP]
        newObjNP = self.addNewObject(objDef.name, parent=parent, fSelectObject=False)
        newObjNP.setPos(obj[OG.OBJ_NP].getPos())
        newObjNP.setHpr(obj[OG.OBJ_NP].getHpr())
        newObjNP.setScale(obj[OG.OBJ_NP].getScale())
        newObj = self.findObjectByNodePath(NodePath(newObjNP))
        if newObj is None:
            return None
        self.updateObjectModel(obj[OG.OBJ_MODEL], newObj, fSelectObject=False)
        self.updateObjectAnim(obj[OG.OBJ_ANIM], newObj, fSelectObject=False)
        for key in obj[OG.OBJ_PROP]:
            self.updateObjectPropValue(newObj, key, obj[OG.OBJ_PROP][key])
        return newObjNP

    def duplicateChild(self, nodePath, parent):
        if False:
            i = 10
            return i + 15
        children = nodePath.findAllMatches('=OBJRoot')
        for childNP in children:
            newChildObjNP = self.duplicateObject(childNP, parent)
            if newChildObjNP is not None:
                self.duplicateChild(childNP, newChildObjNP)

    def duplicateSelected(self):
        if False:
            while True:
                i = 10
        selectedNPs = base.direct.selected.getSelectedAsList()
        duplicatedNPs = []
        for nodePath in selectedNPs:
            newObjNP = self.duplicateObject(nodePath)
            if newObjNP is not None:
                self.duplicateChild(nodePath, newObjNP)
                duplicatedNPs.append(newObjNP)
        base.direct.deselectAllCB()
        for newNodePath in duplicatedNPs:
            base.direct.select(newNodePath, fMultiSelect=1, fUndo=0)
        self.editor.fNeedToSave = True

    def makeSelectedLive(self):
        if False:
            while True:
                i = 10
        obj = self.findObjectByNodePath(base.direct.selected.last)
        if obj:
            if self.currLiveNP:
                self.currLiveNP.clearColorScale()
                if self.currLiveNP == obj[OG.OBJ_NP]:
                    self.currLiveNP = None
                    return
            self.currLiveNP = obj[OG.OBJ_NP]
            self.currLiveNP.setColorScale(0, 1, 0, 1)

    def replaceObjectWithTypeName(self, obj, typeName):
        if False:
            while True:
                i = 10
        uid = obj[OG.OBJ_UID]
        objNP = obj[OG.OBJ_NP]
        mat = objNP.getMat()
        parentObj = self.findObjectByNodePath(objNP.getParent())
        if parentObj:
            parentNP = parentObj[OG.OBJ_NP]
        else:
            parentNP = None
        self.removeObjectById(uid)
        self.editor.ui.sceneGraphUI.delete(uid)
        newobj = self.addNewObject(typeName, uid, parent=parentNP, fSelectObject=False)
        newobj.setMat(mat)

    def flatten(self, newobjModel, model, objDef, uid):
        if False:
            while True:
                i = 10
        pass

    def findActors(self, parent):
        if False:
            for i in range(10):
                print('nop')
        for child in parent.getChildren():
            if child.hasTag('OBJRoot') and (not child.hasTag('Controller')):
                obj = self.findObjectByNodePath(child)
                if obj:
                    if isinstance(obj[OG.OBJ_NP], Actor):
                        self.Actor.append(obj)
                self.findActors(child)

    def findNodes(self, parent):
        if False:
            i = 10
            return i + 15
        for child in parent.getChildren():
            if child.hasTag('OBJRoot') and (not child.hasTag('Controller')):
                obj = self.findObjectByNodePath(child)
                if obj:
                    self.Nodes.append(obj)
                self.findActors(child)