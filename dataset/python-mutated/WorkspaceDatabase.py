from __future__ import absolute_import
from Common.StringUtils import *
from Common.DataType import *
from Common.Misc import *
from types import *
from .MetaDataTable import *
from .MetaFileTable import *
from .MetaFileParser import *
from Workspace.DecBuildData import DecBuildData
from Workspace.DscBuildData import DscBuildData
from Workspace.InfBuildData import InfBuildData

class WorkspaceDatabase(object):

    class BuildObjectFactory(object):
        _FILE_TYPE_ = {'.inf': MODEL_FILE_INF, '.dec': MODEL_FILE_DEC, '.dsc': MODEL_FILE_DSC}
        _FILE_PARSER_ = {MODEL_FILE_INF: InfParser, MODEL_FILE_DEC: DecParser, MODEL_FILE_DSC: DscParser}
        _GENERATOR_ = {MODEL_FILE_INF: InfBuildData, MODEL_FILE_DEC: DecBuildData, MODEL_FILE_DSC: DscBuildData}
        _CACHE_ = {}

        def GetCache(self):
            if False:
                while True:
                    i = 10
            return self._CACHE_

        def __init__(self, WorkspaceDb):
            if False:
                print('Hello World!')
            self.WorkspaceDb = WorkspaceDb

        def __contains__(self, Key):
            if False:
                for i in range(10):
                    print('nop')
            FilePath = Key[0]
            if len(Key) > 1:
                Arch = Key[1]
            else:
                Arch = None
            return (FilePath, Arch) in self._CACHE_

        def __getitem__(self, Key):
            if False:
                i = 10
                return i + 15
            FilePath = Key[0]
            KeyLength = len(Key)
            if KeyLength > 1:
                Arch = Key[1]
            else:
                Arch = None
            if KeyLength > 2:
                Target = Key[2]
            else:
                Target = None
            if KeyLength > 3:
                Toolchain = Key[3]
            else:
                Toolchain = None
            Key = (FilePath, Arch, Target, Toolchain)
            if Key in self._CACHE_:
                return self._CACHE_[Key]
            BuildObject = self.CreateBuildObject(FilePath, Arch, Target, Toolchain)
            self._CACHE_[Key] = BuildObject
            return BuildObject

        def CreateBuildObject(self, FilePath, Arch, Target, Toolchain):
            if False:
                while True:
                    i = 10
            Ext = FilePath.Type
            if Ext not in self._FILE_TYPE_:
                return None
            FileType = self._FILE_TYPE_[Ext]
            if FileType not in self._GENERATOR_:
                return None
            MetaFile = self._FILE_PARSER_[FileType](FilePath, FileType, Arch, MetaFileStorage(self.WorkspaceDb, FilePath, FileType))
            MetaFile.DoPostProcess()
            BuildObject = self._GENERATOR_[FileType](FilePath, MetaFile, self, Arch, Target, Toolchain)
            return BuildObject

    class TransformObjectFactory:

        def __init__(self, WorkspaceDb):
            if False:
                while True:
                    i = 10
            self.WorkspaceDb = WorkspaceDb

        def __getitem__(self, Key):
            if False:
                i = 10
                return i + 15
            pass

    def __init__(self):
        if False:
            while True:
                i = 10
        self.DB = dict()
        self.TblDataModel = DataClass.MODEL_LIST
        self.TblFile = []
        self.Platform = None
        self.BuildObject = WorkspaceDatabase.BuildObjectFactory(self)
        self.TransformObject = WorkspaceDatabase.TransformObjectFactory(self)

    def GetPackageList(self, Platform, Arch, TargetName, ToolChainTag):
        if False:
            return 10
        self.Platform = Platform
        PackageList = []
        Pa = self.BuildObject[self.Platform, Arch, TargetName, ToolChainTag]
        for Module in Pa.Modules:
            ModuleObj = self.BuildObject[Module, Arch, TargetName, ToolChainTag]
            for Package in ModuleObj.Packages:
                if Package not in PackageList:
                    PackageList.append(Package)
        for Lib in Pa.LibraryInstances:
            LibObj = self.BuildObject[Lib, Arch, TargetName, ToolChainTag]
            for Package in LibObj.Packages:
                if Package not in PackageList:
                    PackageList.append(Package)
        for Package in Pa.Packages:
            if Package in PackageList:
                continue
            PackageList.append(Package)
        return PackageList

    def MapPlatform(self, Dscfile):
        if False:
            for i in range(10):
                print('nop')
        Platform = self.BuildObject[PathClass(Dscfile), TAB_COMMON]
        if Platform is None:
            EdkLogger.error('build', PARSER_ERROR, 'Failed to parser DSC file: %s' % Dscfile)
        return Platform
BuildDB = WorkspaceDatabase()
if __name__ == '__main__':
    pass