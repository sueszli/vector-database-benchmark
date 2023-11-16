import imp

class LevelLoaderBase:
    """
    Base calss for LevelLoader

    which you will use to load level editor data in your game.
    Refer LevelLoader.py for example.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.defaultPath = None
        self.initLoader()

    def initLoader(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('populate() must be implemented in your LevelLoader.py')

    def cleanUp(self):
        if False:
            for i in range(10):
                print('nop')
        del base.objectPalette
        del base.protoPalette
        del base.objectHandler
        del base.objectMgr

    def loadFromFile(self, fileName, filePath=None):
        if False:
            return 10
        if filePath is None:
            filePath = self.defaultPath
        if fileName.endswith('.py'):
            fileName = fileName[:-3]
        (file, pathname, description) = imp.find_module(fileName, [filePath])
        try:
            module = imp.load_module(fileName, file, pathname, description)
            return True
        except Exception:
            print('failed to load %s' % fileName)
            return None