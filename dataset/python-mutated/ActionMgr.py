from panda3d.core import Mat4
from direct.showbase.PythonUtil import Functor
from . import ObjectGlobals as OG

class ActionMgr:

    def __init__(self):
        if False:
            return 10
        self.undoList = []
        self.redoList = []

    def reset(self):
        if False:
            print('Hello World!')
        while len(self.undoList) > 0:
            action = self.undoList.pop()
            action.destroy()
        while len(self.redoList) > 0:
            action = self.redoList.pop()
            action.destroy()

    def push(self, action):
        if False:
            return 10
        self.undoList.append(action)
        if len(self.redoList) > 0:
            self.redoList.pop()

    def undo(self):
        if False:
            return 10
        if len(self.undoList) < 1:
            print('No more undo')
        else:
            action = self.undoList.pop()
            self.redoList.append(action)
            action.undo()

    def redo(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.redoList) < 1:
            print('No more redo')
        else:
            action = self.redoList.pop()
            self.undoList.append(action)
            action.redo()

class ActionBase(Functor):
    """ Base class for user actions """

    def __init__(self, function, *args, **kargs):
        if False:
            i = 10
            return i + 15
        self.function = function
        if function is None:

            def nullFunc():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            function = nullFunc
        Functor.__init__(self, function, *args, **kargs)
        self.result = None

    def _do__call__(self, *args, **kargs):
        if False:
            while True:
                i = 10
        self.saveStatus()
        self.result = Functor._do__call__(self, *args, **kargs)
        self.postCall()
        return self.result
    __call__ = _do__call__

    def redo(self):
        if False:
            print('Hello World!')
        self.result = self._do__call__()
        return self.result

    def saveStatus(self):
        if False:
            return 10
        pass

    def postCall(self):
        if False:
            while True:
                i = 10
        pass

    def undo(self):
        if False:
            i = 10
            return i + 15
        print('undo method is not defined for this action')

class ActionAddNewObj(ActionBase):
    """ Action class for adding new object """

    def __init__(self, editor, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        self.editor = editor
        function = self.editor.objectMgr.addNewObject
        ActionBase.__init__(self, function, *args, **kargs)
        self.uid = None

    def postCall(self):
        if False:
            print('Hello World!')
        obj = self.editor.objectMgr.findObjectByNodePath(self.result)
        if obj:
            self.uid = obj[OG.OBJ_UID]

    def redo(self):
        if False:
            i = 10
            return i + 15
        if self.uid is None:
            print("Can't redo this add")
        else:
            self.result = self._do__call__(uid=self.uid)
            return self.result

    def undo(self):
        if False:
            print('Hello World!')
        if self.result is None:
            print("Can't undo this add")
        else:
            print('Undo: addNewObject')
            if self.uid:
                obj = self.editor.objectMgr.findObjectById(self.uid)
            else:
                obj = self.editor.objectMgr.findObjectByNodePath(self.result)
            if obj:
                self.uid = obj[OG.OBJ_UID]
                self.editor.ui.sceneGraphUI.delete(self.uid)
                base.direct.deselect(obj[OG.OBJ_NP])
                base.direct.removeNodePath(obj[OG.OBJ_NP])
                self.result = None
            else:
                print("Can't undo this add")

class ActionDeleteObj(ActionBase):
    """ Action class for deleting object """

    def __init__(self, editor, *args, **kargs):
        if False:
            i = 10
            return i + 15
        self.editor = editor
        function = base.direct.removeAllSelected
        ActionBase.__init__(self, function, *args, **kargs)
        self.selectedUIDs = []
        self.hierarchy = {}
        self.objInfos = {}
        self.objTransforms = {}

    def saveStatus(self):
        if False:
            print('Hello World!')
        selectedNPs = base.direct.selected.getSelectedAsList()

        def saveObjStatus(np, isRecursive=True):
            if False:
                for i in range(10):
                    print('nop')
            obj = self.editor.objectMgr.findObjectByNodePath(np)
            if obj:
                uid = obj[OG.OBJ_UID]
                if not isRecursive:
                    self.selectedUIDs.append(uid)
                objNP = obj[OG.OBJ_NP]
                self.objInfos[uid] = obj
                self.objTransforms[uid] = objNP.getMat()
                parentNP = objNP.getParent()
                if parentNP == render:
                    self.hierarchy[uid] = None
                else:
                    parentObj = self.editor.objectMgr.findObjectByNodePath(parentNP)
                    if parentObj:
                        self.hierarchy[uid] = parentObj[OG.OBJ_UID]
                for child in np.getChildren():
                    if child.hasTag('OBJRoot'):
                        saveObjStatus(child)
        for np in selectedNPs:
            saveObjStatus(np, False)

    def undo(self):
        if False:
            while True:
                i = 10
        if len(self.hierarchy) == 0 or len(self.objInfos) == 0:
            print("Can't undo this deletion")
        else:
            print('Undo: deleteObject')

            def restoreObject(uid, parentNP):
                if False:
                    for i in range(10):
                        print('nop')
                obj = self.objInfos[uid]
                objDef = obj[OG.OBJ_DEF]
                objModel = obj[OG.OBJ_MODEL]
                objProp = obj[OG.OBJ_PROP]
                objRGBA = obj[OG.OBJ_RGBA]
                objNP = self.editor.objectMgr.addNewObject(objDef.name, uid, obj[OG.OBJ_MODEL], parentNP)
                self.editor.objectMgr.updateObjectColor(objRGBA[0], objRGBA[1], objRGBA[2], objRGBA[3], objNP)
                self.editor.objectMgr.updateObjectProperties(objNP, objProp)
                objNP.setMat(self.objTransforms[uid])
            while len(self.hierarchy) > 0:
                for uid in self.hierarchy:
                    if self.hierarchy[uid] is None:
                        parentNP = None
                        restoreObject(uid, parentNP)
                        del self.hierarchy[uid]
                    else:
                        parentObj = self.editor.objectMgr.findObjectById(self.hierarchy[uid])
                        if parentObj:
                            parentNP = parentObj[OG.OBJ_NP]
                            restoreObject(uid, parentNP)
                            del self.hierarchy[uid]
            base.direct.deselectAllCB()
            for uid in self.selectedUIDs:
                obj = self.editor.objectMgr.findObjectById(uid)
                if obj:
                    self.editor.select(obj[OG.OBJ_NP], fMultiSelect=1, fUndo=0)
            self.selecteUIDs = []
            self.hierarchy = {}
            self.objInfos = {}

class ActionDeleteObjById(ActionBase):
    """ Action class for deleting object """

    def __init__(self, editor, uid):
        if False:
            i = 10
            return i + 15
        self.editor = editor
        function = self.editor.objectMgr.removeObjectById
        self.uid = uid
        ActionBase.__init__(self, function, self.uid)
        self.hierarchy = {}
        self.objInfos = {}
        self.objTransforms = {}

    def saveStatus(self):
        if False:
            while True:
                i = 10

        def saveObjStatus(uid_np, isUID=False):
            if False:
                return 10
            if isUID:
                obj = self.editor.objectMgr.findObjectById(uid_np)
            else:
                obj = self.editor.objectMgr.findObjectByNodePath(uid_np)
            if obj:
                uid = obj[OG.OBJ_UID]
                objNP = obj[OG.OBJ_NP]
                self.objInfos[uid] = obj
                self.objTransforms[uid] = objNP.getMat()
                parentNP = objNP.getParent()
                if parentNP == render:
                    self.hierarchy[uid] = None
                else:
                    parentObj = self.editor.objectMgr.findObjectByNodePath(parentNP)
                    if parentObj:
                        self.hierarchy[uid] = parentObj[OG.OBJ_UID]
                for child in objNP.getChildren():
                    if child.hasTag('OBJRoot'):
                        saveObjStatus(child)
        saveObjStatus(self.uid, True)

    def undo(self):
        if False:
            print('Hello World!')
        if len(self.hierarchy) == 0 or len(self.objInfos) == 0:
            print("Can't undo this deletion")
        else:
            print('Undo: deleteObjectById')

            def restoreObject(uid, parentNP):
                if False:
                    i = 10
                    return i + 15
                obj = self.objInfos[uid]
                objDef = obj[OG.OBJ_DEF]
                objModel = obj[OG.OBJ_MODEL]
                objProp = obj[OG.OBJ_PROP]
                objRGBA = obj[OG.OBJ_RGBA]
                objNP = self.editor.objectMgr.addNewObject(objDef.name, uid, obj[OG.OBJ_MODEL], parentNP)
                self.editor.objectMgr.updateObjectColor(objRGBA[0], objRGBA[1], objRGBA[2], objRGBA[3], objNP)
                self.editor.objectMgr.updateObjectProperties(objNP, objProp)
                objNP.setMat(self.objTransforms[uid])
            while len(self.hierarchy) > 0:
                for uid in self.hierarchy:
                    if self.hierarchy[uid] is None:
                        parentNP = None
                        restoreObject(uid, parentNP)
                        del self.hierarchy[uid]
                    else:
                        parentObj = self.editor.objectMgr.findObjectById(self.hierarchy[uid])
                        if parentObj:
                            parentNP = parentObj[OG.OBJ_NP]
                            restoreObject(uid, parentNP)
                            del self.hierarchy[uid]
            self.hierarchy = {}
            self.objInfos = {}

class ActionChangeHierarchy(ActionBase):
    """ Action class for changing Scene Graph Hierarchy """

    def __init__(self, editor, oldGrandParentId, oldParentId, newParentId, childName, *args, **kargs):
        if False:
            i = 10
            return i + 15
        self.editor = editor
        self.oldGrandParentId = oldGrandParentId
        self.oldParentId = oldParentId
        self.newParentId = newParentId
        self.childName = childName
        function = self.editor.ui.sceneGraphUI.parent
        ActionBase.__init__(self, function, self.oldParentId, self.newParentId, self.childName, **kargs)

    def undo(self):
        if False:
            i = 10
            return i + 15
        self.editor.ui.sceneGraphUI.parent(self.oldParentId, self.oldGrandParentId, self.childName)

class ActionSelectObj(ActionBase):
    """ Action class for adding new object """

    def __init__(self, editor, *args, **kargs):
        if False:
            i = 10
            return i + 15
        self.editor = editor
        function = base.direct.selectCB
        ActionBase.__init__(self, function, *args, **kargs)
        self.selectedUIDs = []

    def saveStatus(self):
        if False:
            while True:
                i = 10
        selectedNPs = base.direct.selected.getSelectedAsList()
        for np in selectedNPs:
            obj = self.editor.objectMgr.findObjectByNodePath(np)
            if obj:
                uid = obj[OG.OBJ_UID]
                self.selectedUIDs.append(uid)

    def undo(self):
        if False:
            while True:
                i = 10
        print('Undo : selectObject')
        base.direct.deselectAllCB()
        for uid in self.selectedUIDs:
            obj = self.editor.objectMgr.findObjectById(uid)
            if obj:
                self.editor.select(obj[OG.OBJ_NP], fMultiSelect=1, fUndo=0)
        self.selectedUIDs = []

class ActionTransformObj(ActionBase):
    """ Action class for object transformation """

    def __init__(self, editor, *args, **kargs):
        if False:
            print('Hello World!')
        self.editor = editor
        function = self.editor.objectMgr.setObjectTransform
        ActionBase.__init__(self, function, *args, **kargs)
        self.uid = args[0]
        self.origMat = None

    def saveStatus(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.editor.objectMgr.findObjectById(self.uid)
        if obj:
            self.origMat = Mat4(self.editor.objectMgr.objectsLastXform[obj[OG.OBJ_UID]])

    def _do__call__(self, *args, **kargs):
        if False:
            print('Hello World!')
        self.result = ActionBase._do__call__(self, *args, **kargs)
        obj = self.editor.objectMgr.findObjectById(self.uid)
        if obj:
            self.editor.objectMgr.objectsLastXform[self.uid] = Mat4(obj[OG.OBJ_NP].getMat())
        return self.result

    def undo(self):
        if False:
            for i in range(10):
                print('nop')
        if self.origMat is None:
            print("Can't undo this transform")
        else:
            print('Undo: transformObject')
            obj = self.editor.objectMgr.findObjectById(self.uid)
            if obj:
                obj[OG.OBJ_NP].setMat(self.origMat)
                self.editor.objectMgr.objectsLastXform[self.uid] = Mat4(self.origMat)
            del self.origMat
            self.origMat = None

class ActionDeselectAll(ActionBase):
    """ Action class for adding new object """

    def __init__(self, editor, *args, **kargs):
        if False:
            return 10
        self.editor = editor
        function = base.direct.deselectAllCB
        ActionBase.__init__(self, function, *args, **kargs)
        self.selectedUIDs = []

    def saveStatus(self):
        if False:
            print('Hello World!')
        selectedNPs = base.direct.selected.getSelectedAsList()
        for np in selectedNPs:
            obj = self.editor.objectMgr.findObjectByNodePath(np)
            if obj:
                uid = obj[OG.OBJ_UID]
                self.selectedUIDs.append(uid)

    def undo(self):
        if False:
            return 10
        print('Undo : deselectAll')
        base.direct.deselectAllCB()
        for uid in self.selectedUIDs:
            obj = self.editor.objectMgr.findObjectById(uid)
            if obj:
                self.editor.select(obj[OG.OBJ_NP], fMultiSelect=1, fUndo=0)
        self.selectedUIDs = []

class ActionUpdateObjectProp(ActionBase):
    """ Action class for updating object property """

    def __init__(self, editor, fSelectObject, obj, propName, val, oldVal, function, undoFunc, *args, **kargs):
        if False:
            while True:
                i = 10
        self.editor = editor
        self.fSelectObject = fSelectObject
        self.obj = obj
        self.propName = propName
        self.newVal = val
        self.oldVal = oldVal
        self.undoFunc = undoFunc
        ActionBase.__init__(self, function, *args, **kargs)

    def saveStatus(self):
        if False:
            return 10
        self.obj[OG.OBJ_PROP][self.propName] = self.newVal

    def redo(self):
        if False:
            for i in range(10):
                print('nop')
        self.result = self._do__call__()
        if self.editor and self.fSelectObject:
            base.direct.select(self.obj[OG.OBJ_NP], fUndo=0)
        return self.result

    def undo(self):
        if False:
            return 10
        print('Undo : updateObjectProp')
        if self.oldVal:
            self.obj[OG.OBJ_PROP][self.propName] = self.oldVal
            if self.undoFunc:
                self.undoFunc()
                if self.editor and self.fSelectObject:
                    base.direct.select(self.obj[OG.OBJ_NP], fUndo=0)