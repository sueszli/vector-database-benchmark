"""
GenMetaFileMisc
"""
from Library import DataType as DT
from Library import GlobalData
from Parser.DecParser import Dec

def AddExternToDefineSec(SectionDict, Arch, ExternList):
    if False:
        return 10
    LeftOffset = 31
    for (ArchList, EntryPoint, UnloadImage, Constructor, Destructor, FFE, HelpStringList) in ExternList:
        if Arch or ArchList:
            if EntryPoint:
                Statement = (u'%s ' % DT.TAB_INF_DEFINES_ENTRY_POINT).ljust(LeftOffset) + u'= %s' % EntryPoint
                if FFE:
                    Statement += ' | %s' % FFE
                if len(HelpStringList) > 0:
                    Statement = HelpStringList[0].GetString() + '\n' + Statement
                if len(HelpStringList) > 1:
                    Statement = Statement + HelpStringList[1].GetString()
                SectionDict[Arch] = SectionDict[Arch] + [Statement]
            if UnloadImage:
                Statement = (u'%s ' % DT.TAB_INF_DEFINES_UNLOAD_IMAGE).ljust(LeftOffset) + u'= %s' % UnloadImage
                if FFE:
                    Statement += ' | %s' % FFE
                if len(HelpStringList) > 0:
                    Statement = HelpStringList[0].GetString() + '\n' + Statement
                if len(HelpStringList) > 1:
                    Statement = Statement + HelpStringList[1].GetString()
                SectionDict[Arch] = SectionDict[Arch] + [Statement]
            if Constructor:
                Statement = (u'%s ' % DT.TAB_INF_DEFINES_CONSTRUCTOR).ljust(LeftOffset) + u'= %s' % Constructor
                if FFE:
                    Statement += ' | %s' % FFE
                if len(HelpStringList) > 0:
                    Statement = HelpStringList[0].GetString() + '\n' + Statement
                if len(HelpStringList) > 1:
                    Statement = Statement + HelpStringList[1].GetString()
                SectionDict[Arch] = SectionDict[Arch] + [Statement]
            if Destructor:
                Statement = (u'%s ' % DT.TAB_INF_DEFINES_DESTRUCTOR).ljust(LeftOffset) + u'= %s' % Destructor
                if FFE:
                    Statement += ' | %s' % FFE
                if len(HelpStringList) > 0:
                    Statement = HelpStringList[0].GetString() + '\n' + Statement
                if len(HelpStringList) > 1:
                    Statement = Statement + HelpStringList[1].GetString()
                SectionDict[Arch] = SectionDict[Arch] + [Statement]

def ObtainPcdName(Packages, TokenSpaceGuidValue, Token):
    if False:
        while True:
            i = 10
    TokenSpaceGuidName = ''
    PcdCName = ''
    TokenSpaceGuidNameFound = False
    for PackageDependency in Packages:
        Guid = PackageDependency.GetGuid()
        Version = PackageDependency.GetVersion()
        Path = None
        for PkgInfo in GlobalData.gWSPKG_LIST:
            if Guid == PkgInfo[1]:
                if not Version or Version == PkgInfo[2]:
                    Path = PkgInfo[3]
                    break
        if Path:
            DecFile = None
            if Path not in GlobalData.gPackageDict:
                DecFile = Dec(Path)
                GlobalData.gPackageDict[Path] = DecFile
            else:
                DecFile = GlobalData.gPackageDict[Path]
            DecGuidsDict = DecFile.GetGuidSectionObject().ValueDict
            DecPcdsDict = DecFile.GetPcdSectionObject().ValueDict
            TokenSpaceGuidName = ''
            PcdCName = ''
            TokenSpaceGuidNameFound = False
            for GuidKey in DecGuidsDict:
                GuidList = DecGuidsDict[GuidKey]
                for GuidItem in GuidList:
                    if TokenSpaceGuidValue.upper() == GuidItem.GuidString.upper():
                        TokenSpaceGuidName = GuidItem.GuidCName
                        TokenSpaceGuidNameFound = True
                        break
                if TokenSpaceGuidNameFound:
                    break
            for PcdKey in DecPcdsDict:
                PcdList = DecPcdsDict[PcdKey]
                for PcdItem in PcdList:
                    if TokenSpaceGuidName == PcdItem.TokenSpaceGuidCName and Token == PcdItem.TokenValue:
                        PcdCName = PcdItem.TokenCName
                        return (TokenSpaceGuidName, PcdCName)
        else:
            for Dist in GlobalData.gTO_BE_INSTALLED_DIST_LIST:
                for Package in Dist.PackageSurfaceArea.values():
                    if Guid == Package.Guid:
                        for GuidItem in Package.GuidList:
                            if TokenSpaceGuidValue.upper() == GuidItem.Guid.upper():
                                TokenSpaceGuidName = GuidItem.CName
                                TokenSpaceGuidNameFound = True
                                break
                        for PcdItem in Package.PcdList:
                            if TokenSpaceGuidName == PcdItem.TokenSpaceGuidCName and Token == PcdItem.Token:
                                PcdCName = PcdItem.CName
                                return (TokenSpaceGuidName, PcdCName)
    return (TokenSpaceGuidName, PcdCName)

def TransferDict(OrigDict, Type=None):
    if False:
        print('Hello World!')
    NewDict = {}
    LeftOffset = 0
    if Type in ['INF_GUID', 'INF_PPI_PROTOCOL']:
        LeftOffset = 45
    if Type in ['INF_PCD']:
        LeftOffset = 75
    if LeftOffset > 0:
        for (Statement, SortedArch) in OrigDict:
            if len(Statement) > LeftOffset:
                LeftOffset = len(Statement)
    for (Statement, SortedArch) in OrigDict:
        Comment = OrigDict[Statement, SortedArch]
        if Comment.find('\n') != len(Comment) - 1:
            NewStateMent = Comment + Statement
        elif LeftOffset:
            NewStateMent = Statement.ljust(LeftOffset) + ' ' + Comment.rstrip('\n')
        else:
            NewStateMent = Statement + ' ' + Comment.rstrip('\n')
        if SortedArch in NewDict:
            NewDict[SortedArch] = NewDict[SortedArch] + [NewStateMent]
        else:
            NewDict[SortedArch] = [NewStateMent]
    return NewDict