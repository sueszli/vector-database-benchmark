import copy
from . import ObjectGlobals as OG

class ObjectGen:
    """ Base class for obj definitions """

    def __init__(self, name=''):
        if False:
            while True:
                i = 10
        self.name = name

class ObjectBase(ObjectGen):
    """ Base class for obj definitions """

    def __init__(self, name='', createFunction=None, model=None, models=[], anims=[], animNames=[], animDict={}, properties={}, movable=True, actor=False, named=False, updateModelFunction=None, orderedProperties=[], propertiesMask={}):
        if False:
            while True:
                i = 10
        ObjectGen.__init__(self, name)
        self.createFunction = createFunction
        self.model = model
        self.models = models[:]
        self.anims = anims[:]
        self.animNames = animNames[:]
        self.animDict = copy.deepcopy(animDict)
        self.properties = copy.deepcopy(properties)
        self.movable = movable
        self.actor = actor
        self.named = named
        self.updateModelFunction = updateModelFunction
        self.orderedProperties = orderedProperties[:]
        self.propertiesMask = copy.deepcopy(propertiesMask)

class ObjectCurve(ObjectBase):

    def __init__(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        ObjectBase.__init__(self, *args, **kw)
        self.properties['Degree'] = [OG.PROP_UI_COMBO, OG.PROP_INT, ('base.le.objectMgr.updateCurve', {'val': OG.ARG_VAL, 'obj': OG.ARG_OBJ}), 3, [2, 3, 4]]

class ObjectPaletteBase:
    """
    Base class for objectPalette

    You should write your own ObjectPalette class inheriting this.
    Refer ObjectPalette.py for example.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.rootName = '_root'
        self.data = {}
        self.dataStruct = {}
        self.dataKeys = []
        self.populateSystemObjs()

    def insertItem(self, item, parentName):
        if False:
            i = 10
            return i + 15
        "\n        You can insert item to obj palette tree.\n\n        'item' is the object to be inserted, it can be either a group or obj.\n        'parentName' is the name of parent under where this item will be inserted.\n        "
        if not isinstance(self.data, dict):
            return None
        if parentName is None:
            parentName = self.rootName
        self.dataStruct[item.name] = parentName
        self.data[item.name] = item
        self.dataKeys.append(item.name)

    def add(self, item, parentName=None):
        if False:
            return 10
        if isinstance(item, str):
            self.insertItem(ObjectGen(name=item), parentName)
        else:
            self.insertItem(item, parentName)

    def addHidden(self, item):
        if False:
            return 10
        if hasattr(item, 'name'):
            self.data[item.name] = item

    def deleteStruct(self, name, deleteItems):
        if False:
            for i in range(10):
                print('nop')
        try:
            item = self.data.pop(name)
            for key in list(self.dataStruct.keys()):
                if self.dataStruct[key] == name:
                    node = self.deleteStruct(key, deleteItems)
                    if node is not None:
                        deleteItems[key] = node
            return item
        except Exception:
            return None

    def delete(self, name):
        if False:
            return 10
        try:
            deleteItems = {}
            node = self.deleteStruct(name, deleteItems)
            if node is not None:
                deleteItems[name] = node
            for key in list(deleteItems.keys()):
                item = self.dataStruct.pop(key)
        except Exception:
            return

    def findItem(self, name):
        if False:
            print('Hello World!')
        try:
            item = self.data[name]
        except Exception:
            return None
        return item

    def findChildren(self, name):
        if False:
            while True:
                i = 10
        result = []
        for key in self.dataKeys:
            if self.dataStruct[key] == name:
                result.append(key)
        return result

    def rename(self, oldName, newName):
        if False:
            i = 10
            return i + 15
        if oldName == newName:
            return False
        if newName == '':
            return False
        try:
            for key in list(self.dataStruct.keys()):
                if self.dataStruct[key] == oldName:
                    self.dataStruct[key] = newName
            self.dataStruct[newName] = self.dataStruct.pop(oldName)
            item = self.data.pop(oldName)
            item.name = newName
            self.data[newName] = item
        except Exception:
            return False
        return True

    def populateSystemObjs(self):
        if False:
            i = 10
            return i + 15
        self.addHidden(ObjectCurve(name='__Curve__'))

    def populate(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('populate() must be implemented in ObjectPalette.py')