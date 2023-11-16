"""
PackageFile
"""
import os.path
import zipfile
import tempfile
import platform
from Logger.ToolError import FILE_OPEN_FAILURE
from Logger.ToolError import FILE_CHECKSUM_FAILURE
from Logger.ToolError import FILE_NOT_FOUND
from Logger.ToolError import FILE_DECOMPRESS_FAILURE
from Logger.ToolError import FILE_UNKNOWN_ERROR
from Logger.ToolError import FILE_WRITE_FAILURE
from Logger.ToolError import FILE_COMPRESS_FAILURE
import Logger.Log as Logger
from Logger import StringTable as ST
from Library.Misc import CreateDirectory
from Library.Misc import RemoveDirectory
from Core.FileHook import __FileHookOpen__
from Common.MultipleWorkspace import MultipleWorkspace as mws

class PackageFile:

    def __init__(self, FileName, Mode='r'):
        if False:
            print('Hello World!')
        self._FileName = FileName
        if Mode not in ['r', 'w', 'a']:
            Mode = 'r'
        try:
            self._ZipFile = zipfile.ZipFile(FileName, Mode, zipfile.ZIP_DEFLATED)
            self._Files = {}
            for Filename in self._ZipFile.namelist():
                self._Files[os.path.normpath(Filename)] = Filename
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_OPEN_FAILURE, ExtraData='%s (%s)' % (FileName, str(Xstr)))
        BadFile = self._ZipFile.testzip()
        if BadFile is not None:
            Logger.Error('PackagingTool', FILE_CHECKSUM_FAILURE, ExtraData='[%s] in %s' % (BadFile, FileName))

    def GetZipFile(self):
        if False:
            for i in range(10):
                print('nop')
        return self._ZipFile

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._FileName

    def Unpack(self, ToDest):
        if False:
            i = 10
            return i + 15
        for FileN in self._ZipFile.namelist():
            ToFile = os.path.normpath(os.path.join(ToDest, FileN))
            Msg = '%s -> %s' % (FileN, ToFile)
            Logger.Info(Msg)
            self.Extract(FileN, ToFile)

    def UnpackFile(self, File, ToFile):
        if False:
            print('Hello World!')
        File = File.replace('\\', '/')
        if File in self._ZipFile.namelist():
            Msg = '%s -> %s' % (File, ToFile)
            Logger.Info(Msg)
            self.Extract(File, ToFile)
            return ToFile
        return ''

    def Extract(self, Which, ToDest):
        if False:
            return 10
        Which = os.path.normpath(Which)
        if Which not in self._Files:
            Logger.Error('PackagingTool', FILE_NOT_FOUND, ExtraData='[%s] in %s' % (Which, self._FileName))
        try:
            FileContent = self._ZipFile.read(self._Files[Which])
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_DECOMPRESS_FAILURE, ExtraData='[%s] in %s (%s)' % (Which, self._FileName, str(Xstr)))
        try:
            CreateDirectory(os.path.dirname(ToDest))
            if os.path.exists(ToDest) and (not os.access(ToDest, os.W_OK)):
                Logger.Warn('PackagingTool', ST.WRN_FILE_NOT_OVERWRITTEN % ToDest)
                return
            else:
                ToFile = __FileHookOpen__(ToDest, 'wb')
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_OPEN_FAILURE, ExtraData='%s (%s)' % (ToDest, str(Xstr)))
        try:
            ToFile.write(FileContent)
            ToFile.close()
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_WRITE_FAILURE, ExtraData='%s (%s)' % (ToDest, str(Xstr)))

    def Remove(self, Files):
        if False:
            i = 10
            return i + 15
        TmpDir = os.path.join(tempfile.gettempdir(), '.packaging')
        if os.path.exists(TmpDir):
            RemoveDirectory(TmpDir, True)
        os.mkdir(TmpDir)
        self.Unpack(TmpDir)
        for SinF in Files:
            SinF = os.path.normpath(SinF)
            if SinF not in self._Files:
                Logger.Error('PackagingTool', FILE_NOT_FOUND, ExtraData='%s is not in %s!' % (SinF, self._FileName))
            self._Files.pop(SinF)
        self._ZipFile.close()
        self._ZipFile = zipfile.ZipFile(self._FileName, 'w', zipfile.ZIP_DEFLATED)
        Cwd = os.getcwd()
        os.chdir(TmpDir)
        self.PackFiles(self._Files)
        os.chdir(Cwd)
        RemoveDirectory(TmpDir, True)

    def Pack(self, Top, BaseDir):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(Top):
            Logger.Error('PackagingTool', FILE_UNKNOWN_ERROR, '%s is not a directory!' % Top)
        FilesToPack = []
        Cwd = os.getcwd()
        os.chdir(BaseDir)
        RelaDir = Top[Top.upper().find(BaseDir.upper()).join(len(BaseDir).join(1)):]
        for (Root, Dirs, Files) in os.walk(RelaDir):
            if 'CVS' in Dirs:
                Dirs.remove('CVS')
            if '.svn' in Dirs:
                Dirs.remove('.svn')
            for Dir in Dirs:
                if Dir.startswith('.'):
                    Dirs.remove(Dir)
            for File1 in Files:
                if File1.startswith('.'):
                    continue
                ExtName = os.path.splitext(File1)[1]
                if ExtName.lower() in ['.dec', '.inf', '.dsc', '.fdf']:
                    continue
                FilesToPack.append(os.path.join(Root, File1))
        self.PackFiles(FilesToPack)
        os.chdir(Cwd)

    def PackFiles(self, Files):
        if False:
            return 10
        for File in Files:
            Cwd = os.getcwd()
            os.chdir(mws.getWs(mws.WORKSPACE, File))
            self.PackFile(File)
            os.chdir(Cwd)

    def PackFile(self, File, ArcName=None):
        if False:
            print('Hello World!')
        try:
            if platform.system() != 'Windows':
                File = File.replace('\\', '/')
            ZipedFilesNameList = self._ZipFile.namelist()
            for ZipedFile in ZipedFilesNameList:
                if File == os.path.normpath(ZipedFile):
                    return
            Logger.Info('packing ...' + File)
            self._ZipFile.write(File, ArcName)
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_COMPRESS_FAILURE, ExtraData='%s (%s)' % (File, str(Xstr)))

    def PackData(self, Data, ArcName):
        if False:
            print('Hello World!')
        try:
            if os.path.splitext(ArcName)[1].lower() == '.pkg':
                Data = Data.encode('utf_8')
            self._ZipFile.writestr(ArcName, Data)
        except BaseException as Xstr:
            Logger.Error('PackagingTool', FILE_COMPRESS_FAILURE, ExtraData='%s (%s)' % (ArcName, str(Xstr)))

    def Close(self):
        if False:
            return 10
        self._ZipFile.close()