from __future__ import print_function
import Common.LongFilePathOs as os
import re
import Common.EdkLogger as EdkLogger
import Common.BuildToolError as BuildToolError
import subprocess
import Common.GlobalData as GlobalData
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.Misc import SaveFileOnChange
from Common.DataType import *
FILE_COMMENT_TEMPLATE = '\n## @file\n#\n#  THIS IS AUTO-GENERATED FILE BY BUILD TOOLS AND PLEASE DO NOT MAKE MODIFICATION.\n#\n#  This file lists all VPD informations for a platform collected by build.exe.\n#\n# Copyright (c) 2010 - 2018, Intel Corporation. All rights reserved.<BR>\n# This program and the accompanying materials\n# are licensed and made available under the terms and conditions of the BSD License\n# which accompanies this distribution.  The full text of the license may be found at\n# http://opensource.org/licenses/bsd-license.php\n#\n# THE PROGRAM IS DISTRIBUTED UNDER THE BSD LICENSE ON AN "AS IS" BASIS,\n# WITHOUT WARRANTIES OR REPRESENTATIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.\n#\n\n'

class VpdInfoFile:
    _rVpdPcdLine = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._VpdArray = {}
        self._VpdInfo = {}

    def Add(self, Vpd, skuname, Offset):
        if False:
            for i in range(10):
                print('nop')
        if Vpd is None:
            EdkLogger.error('VpdInfoFile', BuildToolError.ATTRIBUTE_UNKNOWN_ERROR, 'Invalid VPD PCD entry.')
        if not (Offset >= '0' or Offset == TAB_STAR):
            EdkLogger.error('VpdInfoFile', BuildToolError.PARAMETER_INVALID, 'Invalid offset parameter: %s.' % Offset)
        if Vpd.DatumType == TAB_VOID:
            if Vpd.MaxDatumSize <= '0':
                EdkLogger.error('VpdInfoFile', BuildToolError.PARAMETER_INVALID, 'Invalid max datum size for VPD PCD %s.%s' % (Vpd.TokenSpaceGuidCName, Vpd.TokenCName))
        elif Vpd.DatumType in TAB_PCD_NUMERIC_TYPES:
            if not Vpd.MaxDatumSize:
                Vpd.MaxDatumSize = MAX_SIZE_TYPE[Vpd.DatumType]
        elif Vpd.MaxDatumSize <= '0':
            EdkLogger.error('VpdInfoFile', BuildToolError.PARAMETER_INVALID, 'Invalid max datum size for VPD PCD %s.%s' % (Vpd.TokenSpaceGuidCName, Vpd.TokenCName))
        if Vpd not in self._VpdArray:
            self._VpdArray[Vpd] = {}
        self._VpdArray[Vpd].update({skuname: Offset})

    def Write(self, FilePath):
        if False:
            i = 10
            return i + 15
        if not (FilePath is not None or len(FilePath) != 0):
            EdkLogger.error('VpdInfoFile', BuildToolError.PARAMETER_INVALID, 'Invalid parameter FilePath: %s.' % FilePath)
        Content = FILE_COMMENT_TEMPLATE
        Pcds = sorted(self._VpdArray.keys(), key=lambda x: x.TokenCName)
        for Pcd in Pcds:
            i = 0
            PcdTokenCName = Pcd.TokenCName
            for PcdItem in GlobalData.MixedPcd:
                if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                    PcdTokenCName = PcdItem[0]
            for skuname in self._VpdArray[Pcd]:
                PcdValue = str(Pcd.SkuInfoList[skuname].DefaultValue).strip()
                if PcdValue == '':
                    PcdValue = Pcd.DefaultValue
                Content += '%s.%s|%s|%s|%s|%s  \n' % (Pcd.TokenSpaceGuidCName, PcdTokenCName, skuname, str(self._VpdArray[Pcd][skuname]).strip(), str(Pcd.MaxDatumSize).strip(), PcdValue)
                i += 1
        return SaveFileOnChange(FilePath, Content, False)

    def Read(self, FilePath):
        if False:
            for i in range(10):
                print('nop')
        try:
            fd = open(FilePath, 'r')
        except:
            EdkLogger.error('VpdInfoFile', BuildToolError.FILE_OPEN_FAILURE, 'Fail to open file %s for written.' % FilePath)
        Lines = fd.readlines()
        for Line in Lines:
            Line = Line.strip()
            if len(Line) == 0 or Line.startswith('#'):
                continue
            try:
                (PcdName, SkuId, Offset, Size, Value) = Line.split('#')[0].split('|')
                (PcdName, SkuId, Offset, Size, Value) = (PcdName.strip(), SkuId.strip(), Offset.strip(), Size.strip(), Value.strip())
                (TokenSpaceName, PcdTokenName) = PcdName.split('.')
            except:
                EdkLogger.error('BPDG', BuildToolError.PARSER_ERROR, 'Fail to parse VPD information file %s' % FilePath)
            Found = False
            if (TokenSpaceName, PcdTokenName) not in self._VpdInfo:
                self._VpdInfo[TokenSpaceName, PcdTokenName] = {}
            self._VpdInfo[TokenSpaceName, PcdTokenName][SkuId, Offset] = Value
            for VpdObject in self._VpdArray:
                VpdObjectTokenCName = VpdObject.TokenCName
                for PcdItem in GlobalData.MixedPcd:
                    if (VpdObject.TokenCName, VpdObject.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                        VpdObjectTokenCName = PcdItem[0]
                for sku in VpdObject.SkuInfoList:
                    if VpdObject.TokenSpaceGuidCName == TokenSpaceName and VpdObjectTokenCName == PcdTokenName.strip() and (sku == SkuId):
                        if self._VpdArray[VpdObject][sku] == TAB_STAR:
                            if Offset == TAB_STAR:
                                EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'The offset of %s has not been fixed up by third-party BPDG tool.' % PcdName)
                            self._VpdArray[VpdObject][sku] = Offset
                        Found = True
            if not Found:
                EdkLogger.error('BPDG', BuildToolError.PARSER_ERROR, 'Can not find PCD defined in VPD guid file.')

    def GetCount(self):
        if False:
            while True:
                i = 10
        Count = 0
        for OffsetList in self._VpdArray.values():
            Count += len(OffsetList)
        return Count

    def GetOffset(self, vpd):
        if False:
            print('Hello World!')
        if vpd not in self._VpdArray:
            return None
        if len(self._VpdArray[vpd]) == 0:
            return None
        return self._VpdArray[vpd]

    def GetVpdInfo(self, arg):
        if False:
            i = 10
            return i + 15
        (PcdTokenName, TokenSpaceName) = arg
        return [(sku, offset, value) for ((sku, offset), value) in self._VpdInfo.get((TokenSpaceName, PcdTokenName)).items()]

def CallExtenalBPDGTool(ToolPath, VpdFileName):
    if False:
        print('Hello World!')
    assert ToolPath is not None, 'Invalid parameter ToolPath'
    assert VpdFileName is not None and os.path.exists(VpdFileName), 'Invalid parameter VpdFileName'
    OutputDir = os.path.dirname(VpdFileName)
    FileName = os.path.basename(VpdFileName)
    (BaseName, ext) = os.path.splitext(FileName)
    OutputMapFileName = os.path.join(OutputDir, '%s.map' % BaseName)
    OutputBinFileName = os.path.join(OutputDir, '%s.bin' % BaseName)
    try:
        PopenObject = subprocess.Popen(' '.join([ToolPath, '-o', OutputBinFileName, '-m', OutputMapFileName, '-q', '-f', VpdFileName]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    except Exception as X:
        EdkLogger.error('BPDG', BuildToolError.COMMAND_FAILURE, ExtraData=str(X))
    (out, error) = PopenObject.communicate()
    print(out.decode())
    while PopenObject.returncode is None:
        PopenObject.wait()
    if PopenObject.returncode != 0:
        EdkLogger.debug(EdkLogger.DEBUG_1, 'Fail to call BPDG tool', str(error.decode()))
        EdkLogger.error('BPDG', BuildToolError.COMMAND_FAILURE, 'Fail to execute BPDG tool with exit code: %d, the error message is: \n %s' % (PopenObject.returncode, str(error.decode())))
    return PopenObject.returncode