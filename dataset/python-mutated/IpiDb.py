"""
IpiDb
"""
import sqlite3
import os.path
import time
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import UPT_ALREADY_RUNNING_ERROR
from Logger.ToolError import UPT_DB_UPDATE_ERROR
import platform as pf

class IpiDatabase(object):

    def __init__(self, DbPath, Workspace):
        if False:
            i = 10
            return i + 15
        Dir = os.path.dirname(DbPath)
        if not os.path.isdir(Dir):
            os.mkdir(Dir)
        self.Conn = sqlite3.connect(u''.join(DbPath), isolation_level='DEFERRED')
        self.Conn.execute('PRAGMA page_size=4096')
        self.Conn.execute('PRAGMA synchronous=OFF')
        self.Cur = self.Conn.cursor()
        self.DpTable = 'DpInfo'
        self.PkgTable = 'PkgInfo'
        self.ModInPkgTable = 'ModInPkgInfo'
        self.StandaloneModTable = 'StandaloneModInfo'
        self.ModDepexTable = 'ModDepexInfo'
        self.DpFileListTable = 'DpFileListInfo'
        self.DummyTable = 'Dummy'
        self.Workspace = os.path.normpath(Workspace)

    def InitDatabase(self, SkipLock=False):
        if False:
            while True:
                i = 10
        Logger.Verbose(ST.MSG_INIT_IPI_START)
        if not SkipLock:
            try:
                SqlCommand = '\n                create table %s (\n                Dummy TEXT NOT NULL,\n                PRIMARY KEY (Dummy)\n                )' % self.DummyTable
                self.Cur.execute(SqlCommand)
                self.Conn.commit()
            except sqlite3.OperationalError:
                Logger.Error('UPT', UPT_ALREADY_RUNNING_ERROR, ST.ERR_UPT_ALREADY_RUNNING_ERROR)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        DpGuid TEXT NOT NULL,DpVersion TEXT NOT NULL,\n        InstallTime REAL NOT NULL,\n        NewPkgFileName TEXT NOT NULL,\n        PkgFileName TEXT NOT NULL,\n        RePackage TEXT NOT NULL,\n        PRIMARY KEY (DpGuid, DpVersion)\n        )' % self.DpTable
        self.Cur.execute(SqlCommand)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        FilePath TEXT NOT NULL,\n        DpGuid TEXT,\n        DpVersion TEXT,\n        Md5Sum TEXT,\n        PRIMARY KEY (FilePath)\n        )' % self.DpFileListTable
        self.Cur.execute(SqlCommand)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        PackageGuid TEXT NOT NULL,\n        PackageVersion TEXT NOT NULL,\n        InstallTime REAL NOT NULL,\n        DpGuid TEXT,\n        DpVersion TEXT,\n        InstallPath TEXT NOT NULL,\n        PRIMARY KEY (PackageGuid, PackageVersion, InstallPath)\n        )' % self.PkgTable
        self.Cur.execute(SqlCommand)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        ModuleGuid TEXT NOT NULL,\n        ModuleVersion TEXT NOT NULL,\n        ModuleName TEXT NOT NULL,\n        InstallTime REAL NOT NULL,\n        PackageGuid TEXT,\n        PackageVersion TEXT,\n        InstallPath TEXT NOT NULL,\n        PRIMARY KEY (ModuleGuid, ModuleVersion, ModuleName, InstallPath)\n        )' % self.ModInPkgTable
        self.Cur.execute(SqlCommand)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        ModuleGuid TEXT NOT NULL,\n        ModuleVersion TEXT NOT NULL,\n        ModuleName TEXT NOT NULL,\n        InstallTime REAL NOT NULL,\n        DpGuid TEXT,\n        DpVersion TEXT,\n        InstallPath TEXT NOT NULL,\n        PRIMARY KEY (ModuleGuid, ModuleVersion, ModuleName, InstallPath)\n        )' % self.StandaloneModTable
        self.Cur.execute(SqlCommand)
        SqlCommand = '\n        create table IF NOT EXISTS %s (\n        ModuleGuid TEXT NOT NULL,\n        ModuleVersion TEXT NOT NULL,\n        ModuleName TEXT NOT NULL,\n        InstallPath TEXT NOT NULL,\n        DepexGuid TEXT,\n        DepexVersion TEXT\n        )' % self.ModDepexTable
        self.Cur.execute(SqlCommand)
        self.Conn.commit()
        Logger.Verbose(ST.MSG_INIT_IPI_FINISH)

    def RollBack(self):
        if False:
            print('Hello World!')
        self.Conn.rollback()

    def Commit(self):
        if False:
            for i in range(10):
                print('nop')
        self.Conn.commit()

    def AddDPObject(self, DpObj, NewDpPkgFileName, DpPkgFileName, RePackage):
        if False:
            print('Hello World!')
        try:
            for PkgKey in DpObj.PackageSurfaceArea.keys():
                PkgGuid = PkgKey[0]
                PkgVersion = PkgKey[1]
                PkgInstallPath = PkgKey[2]
                self._AddPackage(PkgGuid, PkgVersion, DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), PkgInstallPath)
                PkgObj = DpObj.PackageSurfaceArea[PkgKey]
                for ModKey in PkgObj.GetModuleDict().keys():
                    ModGuid = ModKey[0]
                    ModVersion = ModKey[1]
                    ModName = ModKey[2]
                    ModInstallPath = ModKey[3]
                    ModInstallPath = os.path.normpath(os.path.join(PkgInstallPath, ModInstallPath))
                    self._AddModuleInPackage(ModGuid, ModVersion, ModName, PkgGuid, PkgVersion, ModInstallPath)
                    ModObj = PkgObj.GetModuleDict()[ModKey]
                    for Dep in ModObj.GetPackageDependencyList():
                        DepexGuid = Dep.GetGuid()
                        DepexVersion = Dep.GetVersion()
                        self._AddModuleDepex(ModGuid, ModVersion, ModName, ModInstallPath, DepexGuid, DepexVersion)
                for (FilePath, Md5Sum) in PkgObj.FileList:
                    self._AddDpFilePathList(DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), FilePath, Md5Sum)
            for ModKey in DpObj.ModuleSurfaceArea.keys():
                ModGuid = ModKey[0]
                ModVersion = ModKey[1]
                ModName = ModKey[2]
                ModInstallPath = ModKey[3]
                self._AddStandaloneModule(ModGuid, ModVersion, ModName, DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), ModInstallPath)
                ModObj = DpObj.ModuleSurfaceArea[ModKey]
                for Dep in ModObj.GetPackageDependencyList():
                    DepexGuid = Dep.GetGuid()
                    DepexVersion = Dep.GetVersion()
                    self._AddModuleDepex(ModGuid, ModVersion, ModName, ModInstallPath, DepexGuid, DepexVersion)
                for (Path, Md5Sum) in ModObj.FileList:
                    self._AddDpFilePathList(DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), Path, Md5Sum)
            for (Path, Md5Sum) in DpObj.FileList:
                self._AddDpFilePathList(DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), Path, Md5Sum)
            self._AddDp(DpObj.Header.GetGuid(), DpObj.Header.GetVersion(), NewDpPkgFileName, DpPkgFileName, RePackage)
        except sqlite3.IntegrityError as DetailMsg:
            Logger.Error('UPT', UPT_DB_UPDATE_ERROR, ST.ERR_UPT_DB_UPDATE_ERROR, ExtraData=DetailMsg)

    def _AddDp(self, Guid, Version, NewDpFileName, DistributionFileName, RePackage):
        if False:
            for i in range(10):
                print('nop')
        if Version is None or len(Version.strip()) == 0:
            Version = 'N/A'
        if NewDpFileName is None or len(NewDpFileName.strip()) == 0:
            PkgFileName = 'N/A'
        else:
            PkgFileName = NewDpFileName
        CurrentTime = time.time()
        SqlCommand = "insert into %s values('%s', '%s', %s, '%s', '%s', '%s')" % (self.DpTable, Guid, Version, CurrentTime, PkgFileName, DistributionFileName, str(RePackage).upper())
        self.Cur.execute(SqlCommand)

    def _AddDpFilePathList(self, DpGuid, DpVersion, Path, Md5Sum):
        if False:
            print('Hello World!')
        Path = os.path.normpath(Path)
        if pf.system() == 'Windows':
            if Path.startswith(self.Workspace):
                Path = Path[len(self.Workspace):]
        elif Path.startswith(self.Workspace + os.sep):
            Path = Path[len(self.Workspace) + 1:]
        SqlCommand = "insert into %s values('%s', '%s', '%s', '%s')" % (self.DpFileListTable, Path, DpGuid, DpVersion, Md5Sum)
        self.Cur.execute(SqlCommand)

    def _AddPackage(self, Guid, Version, DpGuid=None, DpVersion=None, Path=''):
        if False:
            return 10
        if Version is None or len(Version.strip()) == 0:
            Version = 'N/A'
        if DpGuid is None or len(DpGuid.strip()) == 0:
            DpGuid = 'N/A'
        if DpVersion is None or len(DpVersion.strip()) == 0:
            DpVersion = 'N/A'
        CurrentTime = time.time()
        SqlCommand = "insert into %s values('%s', '%s', %s, '%s', '%s', '%s')" % (self.PkgTable, Guid, Version, CurrentTime, DpGuid, DpVersion, Path)
        self.Cur.execute(SqlCommand)

    def _AddModuleInPackage(self, Guid, Version, Name, PkgGuid=None, PkgVersion=None, Path=''):
        if False:
            i = 10
            return i + 15
        if Version is None or len(Version.strip()) == 0:
            Version = 'N/A'
        if PkgGuid is None or len(PkgGuid.strip()) == 0:
            PkgGuid = 'N/A'
        if PkgVersion is None or len(PkgVersion.strip()) == 0:
            PkgVersion = 'N/A'
        if os.name == 'posix':
            Path = Path.replace('\\', os.sep)
        else:
            Path = Path.replace('/', os.sep)
        CurrentTime = time.time()
        SqlCommand = "insert into %s values('%s', '%s', '%s', %s, '%s', '%s', '%s')" % (self.ModInPkgTable, Guid, Version, Name, CurrentTime, PkgGuid, PkgVersion, Path)
        self.Cur.execute(SqlCommand)

    def _AddStandaloneModule(self, Guid, Version, Name, DpGuid=None, DpVersion=None, Path=''):
        if False:
            for i in range(10):
                print('nop')
        if Version is None or len(Version.strip()) == 0:
            Version = 'N/A'
        if DpGuid is None or len(DpGuid.strip()) == 0:
            DpGuid = 'N/A'
        if DpVersion is None or len(DpVersion.strip()) == 0:
            DpVersion = 'N/A'
        CurrentTime = time.time()
        SqlCommand = "insert into %s values('%s', '%s', '%s', %s, '%s', '%s', '%s')" % (self.StandaloneModTable, Guid, Version, Name, CurrentTime, DpGuid, DpVersion, Path)
        self.Cur.execute(SqlCommand)

    def _AddModuleDepex(self, Guid, Version, Name, Path, DepexGuid=None, DepexVersion=None):
        if False:
            while True:
                i = 10
        if DepexGuid is None or len(DepexGuid.strip()) == 0:
            DepexGuid = 'N/A'
        if DepexVersion is None or len(DepexVersion.strip()) == 0:
            DepexVersion = 'N/A'
        if os.name == 'posix':
            Path = Path.replace('\\', os.sep)
        else:
            Path = Path.replace('/', os.sep)
        SqlCommand = "insert into %s values('%s', '%s', '%s', '%s', '%s', '%s')" % (self.ModDepexTable, Guid, Version, Name, Path, DepexGuid, DepexVersion)
        self.Cur.execute(SqlCommand)

    def RemoveDpObj(self, DpGuid, DpVersion):
        if False:
            for i in range(10):
                print('nop')
        PkgList = self.GetPackageListFromDp(DpGuid, DpVersion)
        SqlCommand = "delete from ModDepexInfo where ModDepexInfo.ModuleGuid in\n        (select ModuleGuid from StandaloneModInfo as B where B.DpGuid = '%s'\n        and B.DpVersion = '%s')\n        and ModDepexInfo.ModuleVersion in\n        (select ModuleVersion from StandaloneModInfo as B\n        where B.DpGuid = '%s' and B.DpVersion = '%s')\n        and ModDepexInfo.ModuleName in\n        (select ModuleName from StandaloneModInfo as B\n        where B.DpGuid = '%s' and B.DpVersion = '%s')\n        and ModDepexInfo.InstallPath in\n        (select InstallPath from StandaloneModInfo as B\n        where B.DpGuid = '%s' and B.DpVersion = '%s') " % (DpGuid, DpVersion, DpGuid, DpVersion, DpGuid, DpVersion, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        for Pkg in PkgList:
            SqlCommand = "delete from ModDepexInfo where ModDepexInfo.ModuleGuid in\n            (select ModuleGuid from ModInPkgInfo\n            where ModInPkgInfo.PackageGuid ='%s' and\n            ModInPkgInfo.PackageVersion = '%s')\n            and ModDepexInfo.ModuleVersion in\n            (select ModuleVersion from ModInPkgInfo\n            where ModInPkgInfo.PackageGuid ='%s' and\n            ModInPkgInfo.PackageVersion = '%s')\n            and ModDepexInfo.ModuleName in\n            (select ModuleName from ModInPkgInfo\n            where ModInPkgInfo.PackageGuid ='%s' and\n            ModInPkgInfo.PackageVersion = '%s')\n            and ModDepexInfo.InstallPath in\n            (select InstallPath from ModInPkgInfo where\n            ModInPkgInfo.PackageGuid ='%s'\n            and ModInPkgInfo.PackageVersion = '%s')" % (Pkg[0], Pkg[1], Pkg[0], Pkg[1], Pkg[0], Pkg[1], Pkg[0], Pkg[1])
            self.Cur.execute(SqlCommand)
        SqlCommand = "delete from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.StandaloneModTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        for Pkg in PkgList:
            SqlCommand = "delete from %s where %s.PackageGuid ='%s'\n            and %s.PackageVersion = '%s'" % (self.ModInPkgTable, self.ModInPkgTable, Pkg[0], self.ModInPkgTable, Pkg[1])
            self.Cur.execute(SqlCommand)
        SqlCommand = "delete from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.PkgTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        SqlCommand = "delete from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.DpFileListTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        SqlCommand = "delete from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.DpTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)

    def GetDp(self, Guid, Version):
        if False:
            while True:
                i = 10
        if Version is None or len(Version.strip()) == 0:
            Version = 'N/A'
            Logger.Verbose(ST.MSG_GET_DP_INSTALL_LIST)
            (DpGuid, DpVersion) = (Guid, Version)
            SqlCommand = "select * from %s where DpGuid ='%s'" % (self.DpTable, DpGuid)
            self.Cur.execute(SqlCommand)
        else:
            Logger.Verbose(ST.MSG_GET_DP_INSTALL_INFO_START)
            (DpGuid, DpVersion) = (Guid, Version)
            SqlCommand = "select * from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.DpTable, DpGuid, DpVersion)
            self.Cur.execute(SqlCommand)
        DpList = []
        for DpInfo in self.Cur:
            DpGuid = DpInfo[0]
            DpVersion = DpInfo[1]
            InstallTime = DpInfo[2]
            PkgFileName = DpInfo[3]
            DpList.append((DpGuid, DpVersion, InstallTime, PkgFileName))
        Logger.Verbose(ST.MSG_GET_DP_INSTALL_INFO_FINISH)
        return DpList

    def GetDpInstallDirList(self, Guid, Version):
        if False:
            while True:
                i = 10
        SqlCommand = "select InstallPath from PkgInfo where DpGuid = '%s' and DpVersion = '%s'" % (Guid, Version)
        self.Cur.execute(SqlCommand)
        DirList = []
        for Result in self.Cur:
            if Result[0] not in DirList:
                DirList.append(Result[0])
        SqlCommand = "select InstallPath from StandaloneModInfo where DpGuid = '%s' and DpVersion = '%s'" % (Guid, Version)
        self.Cur.execute(SqlCommand)
        for Result in self.Cur:
            if Result[0] not in DirList:
                DirList.append(Result[0])
        return DirList

    def GetDpFileList(self, Guid, Version):
        if False:
            print('Hello World!')
        (DpGuid, DpVersion) = (Guid, Version)
        SqlCommand = "select * from %s where DpGuid ='%s' and DpVersion = '%s'" % (self.DpFileListTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        PathList = []
        for Result in self.Cur:
            Path = Result[0]
            Md5Sum = Result[3]
            PathList.append((os.path.join(self.Workspace, Path), Md5Sum))
        return PathList

    def GetRePkgDict(self):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = 'select * from %s ' % self.DpTable
        self.Cur.execute(SqlCommand)
        DpInfoList = []
        for Result in self.Cur:
            DpInfoList.append(Result)
        FileDict = {}
        for Result in DpInfoList:
            DpGuid = Result[0]
            DpVersion = Result[1]
            NewDpFileName = Result[3]
            RePackage = Result[5]
            if RePackage == 'TRUE':
                RePackage = True
            else:
                RePackage = False
            for FileInfo in self.GetDpFileList(DpGuid, DpVersion):
                PathInfo = FileInfo[0]
                FileDict[PathInfo] = (DpGuid, DpVersion, NewDpFileName, RePackage)
        return FileDict

    def GetDpByName(self, DistributionFile):
        if False:
            while True:
                i = 10
        SqlCommand = "select * from %s where NewPkgFileName = '%s'" % (self.DpTable, DistributionFile)
        self.Cur.execute(SqlCommand)
        for Result in self.Cur:
            DpGuid = Result[0]
            DpVersion = Result[1]
            NewDpFileName = Result[3]
            return (DpGuid, DpVersion, NewDpFileName)
        else:
            return (None, None, None)

    def GetPackage(self, Guid, Version, DpGuid='', DpVersion=''):
        if False:
            return 10
        if DpVersion == '' or DpGuid == '':
            (PackageGuid, PackageVersion) = (Guid, Version)
            SqlCommand = "select * from %s where PackageGuid ='%s'\n            and PackageVersion = '%s'" % (self.PkgTable, PackageGuid, PackageVersion)
            self.Cur.execute(SqlCommand)
        elif Version is None or len(Version.strip()) == 0:
            SqlCommand = "select * from %s where PackageGuid ='%s'" % (self.PkgTable, Guid)
            self.Cur.execute(SqlCommand)
        else:
            (PackageGuid, PackageVersion) = (Guid, Version)
            SqlCommand = "select * from %s where PackageGuid ='%s' and\n            PackageVersion = '%s'\n                            and DpGuid = '%s' and DpVersion = '%s'" % (self.PkgTable, PackageGuid, PackageVersion, DpGuid, DpVersion)
            self.Cur.execute(SqlCommand)
        PkgList = []
        for PkgInfo in self.Cur:
            PkgGuid = PkgInfo[0]
            PkgVersion = PkgInfo[1]
            InstallTime = PkgInfo[2]
            InstallPath = PkgInfo[5]
            PkgList.append((PkgGuid, PkgVersion, InstallTime, DpGuid, DpVersion, InstallPath))
        return PkgList

    def GetModInPackage(self, Guid, Version, Name, Path, PkgGuid='', PkgVersion=''):
        if False:
            i = 10
            return i + 15
        (ModuleGuid, ModuleVersion, ModuleName, InstallPath) = (Guid, Version, Name, Path)
        if PkgVersion == '' or PkgGuid == '':
            SqlCommand = "select * from %s where ModuleGuid ='%s' and\n            ModuleVersion = '%s' and InstallPath = '%s'\n            and ModuleName = '%s'" % (self.ModInPkgTable, ModuleGuid, ModuleVersion, InstallPath, ModuleName)
            self.Cur.execute(SqlCommand)
        else:
            SqlCommand = "select * from %s where ModuleGuid ='%s' and\n            ModuleVersion = '%s' and InstallPath = '%s'\n            and ModuleName = '%s' and PackageGuid ='%s'\n            and PackageVersion = '%s'\n                            " % (self.ModInPkgTable, ModuleGuid, ModuleVersion, InstallPath, ModuleName, PkgGuid, PkgVersion)
            self.Cur.execute(SqlCommand)
        ModList = []
        for ModInfo in self.Cur:
            ModGuid = ModInfo[0]
            ModVersion = ModInfo[1]
            InstallTime = ModInfo[2]
            InstallPath = ModInfo[5]
            ModList.append((ModGuid, ModVersion, InstallTime, PkgGuid, PkgVersion, InstallPath))
        return ModList

    def GetStandaloneModule(self, Guid, Version, Name, Path, DpGuid='', DpVersion=''):
        if False:
            print('Hello World!')
        (ModuleGuid, ModuleVersion, ModuleName, InstallPath) = (Guid, Version, Name, Path)
        if DpGuid == '':
            SqlCommand = "select * from %s where ModuleGuid ='%s' and\n            ModuleVersion = '%s' and InstallPath = '%s'\n            and ModuleName = '%s'" % (self.StandaloneModTable, ModuleGuid, ModuleVersion, InstallPath, ModuleName)
            self.Cur.execute(SqlCommand)
        else:
            SqlCommand = "select * from %s where ModuleGuid ='%s' and\n            ModuleVersion = '%s' and InstallPath = '%s' and ModuleName = '%s' and DpGuid ='%s' and DpVersion = '%s'\n                            " % (self.StandaloneModTable, ModuleGuid, ModuleVersion, ModuleName, InstallPath, DpGuid, DpVersion)
            self.Cur.execute(SqlCommand)
        ModList = []
        for ModInfo in self.Cur:
            ModGuid = ModInfo[0]
            ModVersion = ModInfo[1]
            InstallTime = ModInfo[2]
            InstallPath = ModInfo[5]
            ModList.append((ModGuid, ModVersion, InstallTime, DpGuid, DpVersion, InstallPath))
        return ModList

    def GetSModInsPathListFromDp(self, DpGuid, DpVersion):
        if False:
            print('Hello World!')
        PathList = []
        SqlCommand = "select InstallPath from %s where DpGuid ='%s'\n        and DpVersion = '%s'\n                        " % (self.StandaloneModTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        for Result in self.Cur:
            InstallPath = Result[0]
            PathList.append(InstallPath)
        return PathList

    def GetPackageListFromDp(self, DpGuid, DpVersion):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = "select * from %s where DpGuid ='%s' and\n        DpVersion = '%s' " % (self.PkgTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        PkgList = []
        for PkgInfo in self.Cur:
            PkgGuid = PkgInfo[0]
            PkgVersion = PkgInfo[1]
            InstallPath = PkgInfo[5]
            PkgList.append((PkgGuid, PkgVersion, InstallPath))
        return PkgList

    def GetDpDependentModuleList(self, DpGuid, DpVersion):
        if False:
            for i in range(10):
                print('nop')
        ModList = []
        PkgList = self.GetPackageListFromDp(DpGuid, DpVersion)
        if len(PkgList) > 0:
            return ModList
        for Pkg in PkgList:
            SqlCommand = "select t1.ModuleGuid, t1.ModuleVersion,\n            t1.InstallPath from %s as t1, %s as t2 where\n            t1.ModuleGuid = t2.ModuleGuid and\n            t1.ModuleVersion = t2.ModuleVersion and t2.DepexGuid ='%s'\n            and (t2.DepexVersion = '%s' or t2.DepexVersion = 'N/A') and\n            t1.PackageGuid != '%s' and t1.PackageVersion != '%s'\n                        " % (self.ModInPkgTable, self.ModDepexTable, Pkg[0], Pkg[1], Pkg[0], Pkg[1])
            self.Cur.execute(SqlCommand)
            for ModInfo in self.Cur:
                ModGuid = ModInfo[0]
                ModVersion = ModInfo[1]
                InstallPath = ModInfo[2]
                ModList.append((ModGuid, ModVersion, InstallPath))
            SqlCommand = "select t1.ModuleGuid, t1.ModuleVersion, t1.InstallPath\n            from %s as t1, %s as t2 where t1.ModuleGuid = t2.ModuleGuid and\n            t1.ModuleVersion = t2.ModuleVersion and t2.DepexGuid ='%s'\n            and (t2.DepexVersion = '%s' or t2.DepexVersion = 'N/A') and\n                            t1.DpGuid != '%s' and t1.DpVersion != '%s'\n                        " % (self.StandaloneModTable, self.ModDepexTable, Pkg[0], Pkg[1], DpGuid, DpVersion)
            self.Cur.execute(SqlCommand)
            for ModInfo in self.Cur:
                ModGuid = ModInfo[0]
                ModVersion = ModInfo[1]
                InstallPath = ModInfo[2]
                ModList.append((ModGuid, ModVersion, InstallPath))
        return ModList

    def GetDpModuleList(self, DpGuid, DpVersion):
        if False:
            return 10
        ModList = []
        SqlCommand = "select FilePath\n                        from %s\n                        where DpGuid = '%s' and DpVersion = '%s' and\n                        FilePath like '%%.inf'\n                    " % (self.DpFileListTable, DpGuid, DpVersion)
        self.Cur.execute(SqlCommand)
        for ModuleInfo in self.Cur:
            FilePath = ModuleInfo[0]
            ModList.append(os.path.join(self.Workspace, FilePath))
        return ModList

    def GetModuleDepex(self, Guid, Version, Path):
        if False:
            while True:
                i = 10
        SqlCommand = "select * from %s where ModuleGuid ='%s' and\n        ModuleVersion = '%s' and InstallPath ='%s'\n                            " % (self.ModDepexTable, Guid, Version, Path)
        self.Cur.execute(SqlCommand)
        DepexList = []
        for DepInfo in self.Cur:
            DepexGuid = DepInfo[3]
            DepexVersion = DepInfo[4]
            DepexList.append((DepexGuid, DepexVersion))
        return DepexList

    def InventoryDistInstalled(self):
        if False:
            while True:
                i = 10
        SqlCommand = 'select * from %s ' % self.DpTable
        self.Cur.execute(SqlCommand)
        DpInfoList = []
        for Result in self.Cur:
            DpGuid = Result[0]
            DpVersion = Result[1]
            DpAliasName = Result[3]
            DpFileName = Result[4]
            DpInfoList.append((DpGuid, DpVersion, DpFileName, DpAliasName))
        return DpInfoList

    def CloseDb(self):
        if False:
            i = 10
            return i + 15
        SqlCommand = '\n        drop table IF EXISTS %s\n        ' % self.DummyTable
        self.Cur.execute(SqlCommand)
        self.Conn.commit()
        self.Cur.close()
        self.Conn.close()

    def __ConvertToSqlString(self, StringList):
        if False:
            for i in range(10):
                print('nop')
        if self.DpTable:
            pass
        return list(map(lambda s: s.replace("'", "''"), StringList))