from __future__ import absolute_import
import Common.LongFilePathOs as os
from . import EotGlobalData
from Common.LongFilePathSupport import OpenLongFilePath as open

class Report(object):

    def __init__(self, ReportName='Report.html', FvObj=None, DispatchName=None):
        if False:
            print('Hello World!')
        self.ReportName = ReportName
        self.Op = open(ReportName, 'w+')
        self.DispatchList = None
        if DispatchName:
            self.DispatchList = open(DispatchName, 'w+')
        self.FvObj = FvObj
        self.FfsIndex = 0
        self.PpiIndex = 0
        self.ProtocolIndex = 0
        if EotGlobalData.gMACRO['EFI_SOURCE'] == '':
            EotGlobalData.gMACRO['EFI_SOURCE'] = EotGlobalData.gMACRO['EDK_SOURCE']

    def WriteLn(self, Line):
        if False:
            print('Hello World!')
        self.Op.write('%s\n' % Line)

    def GenerateReport(self):
        if False:
            return 10
        self.GenerateHeader()
        self.GenerateFv()
        self.GenerateTail()
        self.Op.close()
        self.GenerateUnDispatchedList()

    def GenerateUnDispatchedList(self):
        if False:
            return 10
        FvObj = self.FvObj
        EotGlobalData.gOP_UN_DISPATCHED.write('%s\n' % FvObj.Name)
        for Item in FvObj.UnDispatchedFfsDict.keys():
            EotGlobalData.gOP_UN_DISPATCHED.write('%s\n' % FvObj.UnDispatchedFfsDict[Item])

    def GenerateFv(self):
        if False:
            for i in range(10):
                print('nop')
        FvObj = self.FvObj
        Content = '  <tr>\n    <td width="20%%"><strong>Name</strong></td>\n    <td width="60%%"><strong>Guid</strong></td>\n    <td width="20%%"><strong>Size</strong></td>\n  </tr>'
        self.WriteLn(Content)
        for Info in FvObj.BasicInfo:
            FvName = Info[0]
            FvGuid = Info[1]
            FvSize = Info[2]
            Content = '  <tr>\n    <td>%s</td>\n    <td>%s</td>\n    <td>%s</td>\n  </tr>' % (FvName, FvGuid, FvSize)
            self.WriteLn(Content)
        Content = '    <td colspan="3"><table width="100%%"  border="1">\n      <tr>'
        self.WriteLn(Content)
        EotGlobalData.gOP_DISPATCH_ORDER.write('Dispatched:\n')
        for FfsId in FvObj.OrderedFfsDict.keys():
            self.GenerateFfs(FvObj.OrderedFfsDict[FfsId])
        Content = '     </table></td>\n  </tr>'
        self.WriteLn(Content)
        Content = '    <td colspan="3"><table width="100%%"  border="1">\n      <tr>\n        <tr><strong>UnDispatched</strong></tr>'
        self.WriteLn(Content)
        EotGlobalData.gOP_DISPATCH_ORDER.write('\nUnDispatched:\n')
        for FfsId in FvObj.UnDispatchedFfsDict.keys():
            self.GenerateFfs(FvObj.UnDispatchedFfsDict[FfsId])
        Content = '     </table></td>\n  </tr>'
        self.WriteLn(Content)

    def GenerateDepex(self, DepexString):
        if False:
            while True:
                i = 10
        NonGuidList = ['AND', 'OR', 'NOT', 'BEFORE', 'AFTER', 'TRUE', 'FALSE']
        ItemList = DepexString.split(' ')
        DepexString = ''
        for Item in ItemList:
            if Item not in NonGuidList:
                SqlCommand = "select DISTINCT GuidName from Report where GuidValue like '%s' and ItemMode = 'Produced' group by GuidName" % Item
                RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
                if RecordSet != []:
                    Item = RecordSet[0][0]
            DepexString = DepexString + Item + ' '
        Content = '                <tr>\n                  <td width="5%%"></td>\n                  <td width="95%%">%s</td>\n                </tr>' % DepexString
        self.WriteLn(Content)

    def GeneratePpi(self, Name, Guid, Type):
        if False:
            for i in range(10):
                print('nop')
        self.GeneratePpiProtocol('Ppi', Name, Guid, Type, self.PpiIndex)

    def GenerateProtocol(self, Name, Guid, Type):
        if False:
            print('Hello World!')
        self.GeneratePpiProtocol('Protocol', Name, Guid, Type, self.ProtocolIndex)

    def GeneratePpiProtocol(self, Model, Name, Guid, Type, CName):
        if False:
            return 10
        Content = '                <tr>\n                  <td width="5%%"></td>\n                  <td width="10%%">%s</td>\n                  <td width="85%%" colspan="3">%s</td>\n                  <!-- %s -->\n                </tr>' % (Model, Name, Guid)
        self.WriteLn(Content)
        if Type == 'Produced':
            SqlCommand = "select DISTINCT SourceFileFullPath, BelongsToFunction from Report where GuidName like '%s' and ItemMode = 'Callback'" % Name
            RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
            for Record in RecordSet:
                SqlCommand = "select FullPath from File\n                                where ID = (\n                                select DISTINCT BelongsToFile from Inf\n                                where Value1 like '%s')" % Record[0]
                ModuleSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
                Inf = ModuleSet[0][0].replace(EotGlobalData.gMACRO['WORKSPACE'], '.')
                Function = Record[1]
                Address = ''
                for Item in EotGlobalData.gMap:
                    if Function in EotGlobalData.gMap[Item]:
                        Address = EotGlobalData.gMap[Item][Function]
                        break
                    if '_' + Function in EotGlobalData.gMap[Item]:
                        Address = EotGlobalData.gMap[Item]['_' + Function]
                        break
                Content = '                <tr>\n                      <td width="5%%"></td>\n                      <td width="10%%">%s</td>\n                      <td width="40%%">%s</td>\n                      <td width="35%%">%s</td>\n                      <td width="10%%">%s</td>\n                    </tr>' % ('Callback', Inf, Function, Address)
                self.WriteLn(Content)

    def GenerateFfs(self, FfsObj):
        if False:
            i = 10
            return i + 15
        self.FfsIndex = self.FfsIndex + 1
        if FfsObj is not None and FfsObj.Type in [3, 4, 5, 6, 7, 8, 10]:
            FfsGuid = FfsObj.Guid
            FfsOffset = FfsObj._OFF_
            FfsName = 'Unknown-Module'
            FfsPath = FfsGuid
            FfsType = FfsObj._TypeName[FfsObj.Type]
            if FfsGuid.upper() == '7BB28B99-61BB-11D5-9A5D-0090273FC14D':
                FfsName = 'Logo'
            if FfsGuid.upper() == '7E374E25-8E01-4FEE-87F2-390C23C606CD':
                FfsName = 'AcpiTables'
            if FfsGuid.upper() == '961578FE-B6B7-44C3-AF35-6BC705CD2B1F':
                FfsName = 'Fat'
            SqlCommand = "select Value2 from Inf\n                            where BelongsToFile = (select BelongsToFile from Inf where Value1 = 'FILE_GUID' and lower(Value2) = lower('%s') and Model = %s)\n                            and Model = %s and Value1='BASE_NAME'" % (FfsGuid, 5001, 5001)
            RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
            if RecordSet != []:
                FfsName = RecordSet[0][0]
            SqlCommand = "select FullPath from File\n                            where ID = (select BelongsToFile from Inf where Value1 = 'FILE_GUID' and lower(Value2) = lower('%s') and Model = %s)\n                            and Model = %s" % (FfsGuid, 5001, 1011)
            RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
            if RecordSet != []:
                FfsPath = RecordSet[0][0]
            Content = '  <tr>\n      <tr class=\'styleFfs\' id=\'FfsHeader%s\'>\n        <td width="55%%"><span onclick="Display(\'FfsHeader%s\', \'Ffs%s\')" onMouseOver="funOnMouseOver()" onMouseOut="funOnMouseOut()">%s</span></td>\n        <td width="15%%">%s</td>\n        <!--<td width="20%%">%s</td>-->\n        <!--<td width="20%%">%s</td>-->\n        <td width="10%%">%s</td>\n      </tr>\n      <tr id=\'Ffs%s\' style=\'display:none;\'>\n        <td colspan="4"><table width="100%%"  border="1">' % (self.FfsIndex, self.FfsIndex, self.FfsIndex, FfsPath, FfsName, FfsGuid, FfsOffset, FfsType, self.FfsIndex)
            if self.DispatchList:
                if FfsObj.Type in [4, 6]:
                    self.DispatchList.write('%s %s %s %s\n' % (FfsGuid, 'P', FfsName, FfsPath))
                if FfsObj.Type in [5, 7, 8, 10]:
                    self.DispatchList.write('%s %s %s %s\n' % (FfsGuid, 'D', FfsName, FfsPath))
            self.WriteLn(Content)
            EotGlobalData.gOP_DISPATCH_ORDER.write('%s\n' % FfsName)
            if FfsObj.Depex != '':
                Content = '          <tr>\n            <td><span id=\'DepexHeader%s\' class="styleDepex" onclick="Display(\'DepexHeader%s\', \'Depex%s\')" onMouseOver="funOnMouseOver()" onMouseOut="funOnMouseOut()">&nbsp&nbspDEPEX expression</span></td>\n          </tr>\n          <tr id=\'Depex%s\' style=\'display:none;\'>\n            <td><table width="100%%"  border="1">' % (self.FfsIndex, self.FfsIndex, self.FfsIndex, self.FfsIndex)
                self.WriteLn(Content)
                self.GenerateDepex(FfsObj.Depex)
                Content = '            </table></td>\n          </tr>'
                self.WriteLn(Content)
            SqlCommand = "select ModuleName, ItemType, GuidName, GuidValue, GuidMacro from Report\n                            where SourceFileFullPath in\n                            (select Value1 from Inf where BelongsToFile =\n                            (select BelongsToFile from Inf\n                            where Value1 = 'FILE_GUID' and Value2 like '%s' and Model = %s)\n                            and Model = %s)\n                            and ItemMode = 'Consumed' group by GuidName order by ItemType" % (FfsGuid, 5001, 3007)
            RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
            if RecordSet != []:
                Count = len(RecordSet)
                Content = '          <tr>\n            <td><span id=\'ConsumedHeader%s\' class="styleConsumed" onclick="Display(\'ConsumedHeader%s\', \'Consumed%s\')" onMouseOver="funOnMouseOver()" onMouseOut="funOnMouseOut()">&nbsp&nbspConsumed Ppis/Protocols List (%s)</span></td>\n          </tr>\n          <tr id=\'Consumed%s\' style=\'display:none;\'>\n            <td><table width="100%%"  border="1">' % (self.FfsIndex, self.FfsIndex, self.FfsIndex, Count, self.FfsIndex)
                self.WriteLn(Content)
                self.ProtocolIndex = 0
                for Record in RecordSet:
                    self.ProtocolIndex = self.ProtocolIndex + 1
                    Name = Record[2]
                    CName = Record[4]
                    Guid = Record[3]
                    Type = Record[1]
                    self.GeneratePpiProtocol(Type, Name, Guid, 'Consumed', CName)
                Content = '            </table></td>\n          </tr>'
                self.WriteLn(Content)
            SqlCommand = "select ModuleName, ItemType, GuidName, GuidValue, GuidMacro from Report\n                            where SourceFileFullPath in\n                            (select Value1 from Inf where BelongsToFile =\n                            (select BelongsToFile from Inf\n                            where Value1 = 'FILE_GUID' and Value2 like '%s' and Model = %s)\n                            and Model = %s)\n                            and ItemMode = 'Produced' group by GuidName order by ItemType" % (FfsGuid, 5001, 3007)
            RecordSet = EotGlobalData.gDb.TblReport.Exec(SqlCommand)
            if RecordSet != []:
                Count = len(RecordSet)
                Content = '          <tr>\n            <td><span id=\'ProducedHeader%s\' class="styleProduced" onclick="Display(\'ProducedHeader%s\', \'Produced%s\')" onMouseOver="funOnMouseOver()" onMouseOut="funOnMouseOut()">&nbsp&nbspProduced Ppis/Protocols List (%s)</span></td>\n          </tr>\n          <tr id=\'Produced%s\' style=\'display:none;\'>\n            <td><table width="100%%"  border="1">' % (self.FfsIndex, self.FfsIndex, self.FfsIndex, Count, self.FfsIndex)
                self.WriteLn(Content)
                self.PpiIndex = 0
                for Record in RecordSet:
                    self.PpiIndex = self.PpiIndex + 1
                    Name = Record[2]
                    CName = Record[4]
                    Guid = Record[3]
                    Type = Record[1]
                    self.GeneratePpiProtocol(Type, Name, Guid, 'Produced', CName)
                Content = '            </table></td>\n          </tr>'
                self.WriteLn(Content)
            RecordSet = None
            Content = '        </table></td>\n        </tr>'
            self.WriteLn(Content)

    def GenerateTail(self):
        if False:
            for i in range(10):
                print('nop')
        Tail = '</table>\n</body>\n</html>'
        self.WriteLn(Tail)

    def GenerateHeader(self):
        if False:
            i = 10
            return i + 15
        Header = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"\n"http://www.w3.org/TR/html4/loose.dtd">\n<html>\n<head>\n<title>Execution Order Tool Report</title>\n<meta http-equiv="Content-Type" content="text/html">\n<style type="text/css">\n<!--\n.styleFfs {\n    color: #006600;\n    font-weight: bold;\n}\n.styleDepex {\n    color: #FF0066;\n    font-weight: bold;\n}\n.styleProduced {\n    color: #0000FF;\n    font-weight: bold;\n}\n.styleConsumed {\n    color: #FF00FF;\n    font-weight: bold;\n}\n-->\n</style>\n<Script type="text/javascript">\nfunction Display(ParentID, SubID)\n{\n    SubItem = document.getElementById(SubID);\n    ParentItem = document.getElementById(ParentID);\n    if (SubItem.style.display == \'none\')\n    {\n        SubItem.style.display = \'\'\n        ParentItem.style.fontWeight = \'normal\'\n    }\n    else\n    {\n        SubItem.style.display = \'none\'\n        ParentItem.style.fontWeight = \'bold\'\n    }\n\n}\n\nfunction funOnMouseOver()\n{\n    document.body.style.cursor = "hand";\n}\n\nfunction funOnMouseOut()\n{\n    document.body.style.cursor = "";\n}\n\n</Script>\n</head>\n\n<body>\n<table width="100%%"  border="1">'
        self.WriteLn(Header)
if __name__ == '__main__':
    FilePath = 'FVRECOVERYFLOPPY.fv'
    if FilePath.lower().endswith('.fv'):
        fd = open(FilePath, 'rb')
        buf = array('B')
        try:
            buf.fromfile(fd, os.path.getsize(FilePath))
        except EOFError:
            pass
        fv = FirmwareVolume('FVRECOVERY', buf, 0)
    report = Report('Report.html', fv)
    report.GenerateReport()