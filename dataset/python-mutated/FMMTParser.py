from FirmwareStorageFormat.Common import *
from core.BinaryFactoryProduct import ParserEntry
from core.BiosTreeNode import *
from core.BiosTree import *
from core.GuidTools import *
from utils.FmmtLogger import FmmtLogger as logger

class FMMTParser:

    def __init__(self, name: str, TYPE: str) -> None:
        if False:
            return 10
        self.WholeFvTree = BIOSTREE(name)
        self.WholeFvTree.type = TYPE
        self.FinalData = b''
        self.BinaryInfo = []

    def ParserFromRoot(self, WholeFvTree=None, whole_data: bytes=b'', Reloffset: int=0) -> None:
        if False:
            print('Hello World!')
        if WholeFvTree.type == ROOT_TREE or WholeFvTree.type == ROOT_FV_TREE:
            ParserEntry().DataParser(self.WholeFvTree, whole_data, Reloffset)
        else:
            ParserEntry().DataParser(WholeFvTree, whole_data, Reloffset)
        for Child in WholeFvTree.Child:
            self.ParserFromRoot(Child, '')

    def Encapsulation(self, rootTree, CompressStatus: bool) -> None:
        if False:
            return 10
        if rootTree.type == ROOT_TREE or rootTree.type == ROOT_FV_TREE or rootTree.type == ROOT_FFS_TREE or (rootTree.type == ROOT_SECTION_TREE):
            logger.debug('Encapsulated successfully!')
        elif rootTree.type == BINARY_DATA or rootTree.type == FFS_FREE_SPACE:
            self.FinalData += rootTree.Data.Data
            rootTree.Child = []
        elif rootTree.type == DATA_FV_TREE or rootTree.type == FFS_PAD:
            self.FinalData += struct2stream(rootTree.Data.Header) + rootTree.Data.Data + rootTree.Data.PadData
            if rootTree.isFinalChild():
                ParTree = rootTree.Parent
                if ParTree.type != 'ROOT':
                    self.FinalData += ParTree.Data.PadData
            rootTree.Child = []
        elif rootTree.type == FV_TREE or rootTree.type == FFS_TREE or rootTree.type == SEC_FV_TREE:
            if rootTree.HasChild():
                self.FinalData += struct2stream(rootTree.Data.Header)
            else:
                self.FinalData += struct2stream(rootTree.Data.Header) + rootTree.Data.Data + rootTree.Data.PadData
                if rootTree.isFinalChild():
                    ParTree = rootTree.Parent
                    if ParTree.type != 'ROOT':
                        self.FinalData += ParTree.Data.PadData
        elif rootTree.type == SECTION_TREE:
            if rootTree.Data.OriData == b'' or (rootTree.Data.OriData != b'' and CompressStatus):
                if rootTree.HasChild():
                    if rootTree.Data.ExtHeader:
                        self.FinalData += struct2stream(rootTree.Data.Header) + struct2stream(rootTree.Data.ExtHeader)
                    else:
                        self.FinalData += struct2stream(rootTree.Data.Header)
                else:
                    Data = rootTree.Data.Data
                    if rootTree.Data.ExtHeader:
                        self.FinalData += struct2stream(rootTree.Data.Header) + struct2stream(rootTree.Data.ExtHeader) + Data + rootTree.Data.PadData
                    else:
                        self.FinalData += struct2stream(rootTree.Data.Header) + Data + rootTree.Data.PadData
                    if rootTree.isFinalChild():
                        ParTree = rootTree.Parent
                        self.FinalData += ParTree.Data.PadData
            else:
                Data = rootTree.Data.OriData
                rootTree.Child = []
                if rootTree.Data.ExtHeader:
                    self.FinalData += struct2stream(rootTree.Data.Header) + struct2stream(rootTree.Data.ExtHeader) + Data + rootTree.Data.PadData
                else:
                    self.FinalData += struct2stream(rootTree.Data.Header) + Data + rootTree.Data.PadData
                if rootTree.isFinalChild():
                    ParTree = rootTree.Parent
                    self.FinalData += ParTree.Data.PadData
        for Child in rootTree.Child:
            self.Encapsulation(Child, CompressStatus)