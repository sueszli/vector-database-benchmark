import sys
import re
import datetime
import os
import logging
from edk2toollib.windows.capsule.cat_generator import CatGenerator
from edk2toollib.windows.capsule.inf_generator import InfGenerator
from edk2toollib.utility_functions import CatalogSignWithSignTool
from edk2toollib.windows.locate_tools import FindToolInWinSdk

class WindowsCapsuleSupportHelper(object):

    def RegisterHelpers(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fp = os.path.abspath(__file__)
        obj.Register('PackageWindowsCapsuleFiles', WindowsCapsuleSupportHelper.PackageWindowsCapsuleFiles, fp)

    @staticmethod
    def PackageWindowsCapsuleFiles(OutputFolder, ProductName, ProductFmpGuid, CapsuleVersion_DotString, CapsuleVersion_HexString, ProductFwProvider, ProductFwMfgName, ProductFwDesc, CapsuleFileName, PfxFile=None, PfxPass=None, Rollback=False, Arch='amd64', OperatingSystem_String='Win10'):
        if False:
            while True:
                i = 10
        logging.debug('CapsulePackage: Create Windows Capsule Files')
        InfFilePath = os.path.join(OutputFolder, ProductName + '.inf')
        InfTool = InfGenerator(ProductName, ProductFwProvider, ProductFmpGuid, Arch, ProductFwDesc, CapsuleVersion_DotString, CapsuleVersion_HexString)
        InfTool.Manufacturer = ProductFwMfgName
        ret = InfTool.MakeInf(InfFilePath, CapsuleFileName, Rollback)
        if ret != 0:
            raise Exception('CreateWindowsInf Failed with errorcode %d' % ret)
        CatFilePath = os.path.realpath(os.path.join(OutputFolder, ProductName + '.cat'))
        CatTool = CatGenerator(Arch, OperatingSystem_String)
        ret = CatTool.MakeCat(CatFilePath)
        if ret != 0:
            raise Exception('Creating Cat file Failed with errorcode %d' % ret)
        if PfxFile is not None:
            SignToolPath = FindToolInWinSdk('signtool.exe')
            if not os.path.exists(SignToolPath):
                raise Exception("Can't find signtool on this machine.")
            ret = CatalogSignWithSignTool(SignToolPath, CatFilePath, PfxFile, PfxPass)
            if ret != 0:
                raise Exception('Signing Cat file Failed with errorcode %d' % ret)
        return ret