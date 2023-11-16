import Common.LongFilePathOs as os
from Common.DataType import TAB_WORKSPACE

class MultipleWorkspace(object):
    WORKSPACE = ''
    PACKAGES_PATH = None

    @classmethod
    def convertPackagePath(cls, Ws, Path):
        if False:
            print('Hello World!')
        if str(os.path.normcase(Path)).startswith(Ws):
            return os.path.join(Ws, os.path.relpath(Path, Ws))
        return Path

    @classmethod
    def setWs(cls, Ws, PackagesPath=None):
        if False:
            print('Hello World!')
        cls.WORKSPACE = Ws
        if PackagesPath:
            cls.PACKAGES_PATH = [cls.convertPackagePath(Ws, os.path.normpath(Path.strip())) for Path in PackagesPath.split(os.pathsep)]
        else:
            cls.PACKAGES_PATH = []

    @classmethod
    def join(cls, Ws, *p):
        if False:
            for i in range(10):
                print('nop')
        Path = os.path.join(Ws, *p)
        if not os.path.exists(Path):
            for Pkg in cls.PACKAGES_PATH:
                Path = os.path.join(Pkg, *p)
                if os.path.exists(Path):
                    return Path
            Path = os.path.join(Ws, *p)
        return Path

    @classmethod
    def relpath(cls, Path, Ws):
        if False:
            return 10
        for Pkg in cls.PACKAGES_PATH:
            if Path.lower().startswith(Pkg.lower()):
                Path = os.path.relpath(Path, Pkg)
                return Path
        if Path.lower().startswith(Ws.lower()):
            Path = os.path.relpath(Path, Ws)
        return Path

    @classmethod
    def getWs(cls, Ws, Path):
        if False:
            for i in range(10):
                print('nop')
        absPath = os.path.join(Ws, Path)
        if not os.path.exists(absPath):
            for Pkg in cls.PACKAGES_PATH:
                absPath = os.path.join(Pkg, Path)
                if os.path.exists(absPath):
                    return Pkg
        return Ws

    @classmethod
    def handleWsMacro(cls, PathStr):
        if False:
            print('Hello World!')
        if TAB_WORKSPACE in PathStr:
            PathList = PathStr.split()
            if PathList:
                for (i, str) in enumerate(PathList):
                    MacroStartPos = str.find(TAB_WORKSPACE)
                    if MacroStartPos != -1:
                        Substr = str[MacroStartPos:]
                        Path = Substr.replace(TAB_WORKSPACE, cls.WORKSPACE).strip()
                        if not os.path.exists(Path):
                            for Pkg in cls.PACKAGES_PATH:
                                Path = Substr.replace(TAB_WORKSPACE, Pkg).strip()
                                if os.path.exists(Path):
                                    break
                        PathList[i] = str[0:MacroStartPos] + Path
            PathStr = ' '.join(PathList)
        return PathStr

    @classmethod
    def getPkgPath(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.PACKAGES_PATH