import os
from core.BiosTree import *
from core.GuidTools import GUIDTools
from core.BiosTreeNode import *
from FirmwareStorageFormat.Common import *
from utils.FmmtLogger import FmmtLogger as logger
EFI_FVB2_ERASE_POLARITY = 2048

def ChangeSize(TargetTree, size_delta: int=0) -> None:
    if False:
        return 10
    if type(TargetTree.Data.Header) == type(EFI_FFS_FILE_HEADER2()) or type(TargetTree.Data.Header) == type(EFI_COMMON_SECTION_HEADER2()):
        TargetTree.Data.Size -= size_delta
        TargetTree.Data.Header.ExtendedSize -= size_delta
    elif TargetTree.type == SECTION_TREE and TargetTree.Data.OriData:
        OriSize = TargetTree.Data.Header.SECTION_SIZE
        OriSize -= size_delta
        TargetTree.Data.Header.Size[0] = OriSize % 16 ** 2
        TargetTree.Data.Header.Size[1] = OriSize % 16 ** 4 // 16 ** 2
        TargetTree.Data.Header.Size[2] = OriSize // 16 ** 4
    else:
        TargetTree.Data.Size -= size_delta
        TargetTree.Data.Header.Size[0] = TargetTree.Data.Size % 16 ** 2
        TargetTree.Data.Header.Size[1] = TargetTree.Data.Size % 16 ** 4 // 16 ** 2
        TargetTree.Data.Header.Size[2] = TargetTree.Data.Size // 16 ** 4

def ModifyFfsType(TargetFfs) -> None:
    if False:
        i = 10
        return i + 15
    if type(TargetFfs.Data.Header) == type(EFI_FFS_FILE_HEADER()) and TargetFfs.Data.Size > 16777215:
        ExtendSize = TargetFfs.Data.Header.FFS_FILE_SIZE + 8
        New_Header = EFI_FFS_FILE_HEADER2()
        New_Header.Name = TargetFfs.Data.Header.Name
        New_Header.IntegrityCheck = TargetFfs.Data.Header.IntegrityCheck
        New_Header.Type = TargetFfs.Data.Header.Type
        New_Header.Attributes = TargetFfs.Data.Header.Attributes | 1
        NewSize = 0
        New_Header.Size[0] = NewSize % 16 ** 2
        New_Header.Size[1] = NewSize % 16 ** 4 // 16 ** 2
        New_Header.Size[2] = NewSize // 16 ** 4
        New_Header.State = TargetFfs.Data.Header.State
        New_Header.ExtendedSize = ExtendSize
        TargetFfs.Data.Header = New_Header
        TargetFfs.Data.Size = TargetFfs.Data.Header.FFS_FILE_SIZE
        TargetFfs.Data.HeaderLength = TargetFfs.Data.Header.HeaderLength
        TargetFfs.Data.ModCheckSum()
    elif type(TargetFfs.Data.Header) == type(EFI_FFS_FILE_HEADER2()) and TargetFfs.Data.Size <= 16777215:
        New_Header = EFI_FFS_FILE_HEADER()
        New_Header.Name = TargetFfs.Data.Header.Name
        New_Header.IntegrityCheck = TargetFfs.Data.Header.IntegrityCheck
        New_Header.Type = TargetFfs.Data.Header.Type
        New_Header.Attributes = TargetFfs.Data.Header.Attributes - 1
        New_Header.Size[0] = (TargetFfs.Data.Size - 8) % 16 ** 2
        New_Header.Size[1] = (TargetFfs.Data.Size - 8) % 16 ** 4 // 16 ** 2
        New_Header.Size[2] = (TargetFfs.Data.Size - 8) // 16 ** 4
        New_Header.State = TargetFfs.Data.Header.State
        TargetFfs.Data.Header = New_Header
        TargetFfs.Data.Size = TargetFfs.Data.Header.FFS_FILE_SIZE
        TargetFfs.Data.HeaderLength = TargetFfs.Data.Header.HeaderLength
        TargetFfs.Data.ModCheckSum()
        if struct2stream(TargetFfs.Parent.Data.Header.FileSystemGuid) == EFI_FIRMWARE_FILE_SYSTEM3_GUID_BYTE:
            NeedChange = True
            for item in TargetFfs.Parent.Child:
                if type(item.Data.Header) == type(EFI_FFS_FILE_HEADER2()):
                    NeedChange = False
            if NeedChange:
                TargetFfs.Parent.Data.Header.FileSystemGuid = ModifyGuidFormat('8c8ce578-8a3d-4f1c-9935-896185c32dd3')
    if type(TargetFfs.Data.Header) == type(EFI_FFS_FILE_HEADER2()):
        TarParent = TargetFfs.Parent
        while TarParent:
            if TarParent.type == FV_TREE and struct2stream(TarParent.Data.Header.FileSystemGuid) == EFI_FIRMWARE_FILE_SYSTEM2_GUID_BYTE:
                TarParent.Data.Header.FileSystemGuid = ModifyGuidFormat('5473C07A-3DCB-4dca-BD6F-1E9689E7349A')
            TarParent = TarParent.Parent

def PadSectionModify(PadSection, Offset) -> None:
    if False:
        print('Hello World!')
    ChangeSize(PadSection, Offset)
    PadSection.Data.Data = (PadSection.Data.Size - PadSection.Data.HeaderLength) * b'\xff'

def ModifySectionType(TargetSection) -> None:
    if False:
        for i in range(10):
            print('nop')
    if type(TargetSection.Data.Header) == type(EFI_COMMON_SECTION_HEADER()) and TargetSection.Data.Size >= 16777215:
        New_Header = EFI_COMMON_SECTION_HEADER2()
        New_Header.Type = TargetSection.Data.Header.Type
        NewSize = 16777215
        New_Header.Size[0] = NewSize % 16 ** 2
        New_Header.Size[1] = NewSize % 16 ** 4 // 16 ** 2
        New_Header.Size[2] = NewSize // 16 ** 4
        New_Header.ExtendedSize = TargetSection.Data.Size + 4
        TargetSection.Data.Header = New_Header
        TargetSection.Data.Size = TargetSection.Data.Header.SECTION_SIZE
        if TargetSection.LastRel.Data.IsPadSection:
            PadSectionModify(TargetSection.LastRel, -4)
        else:
            SecParent = TargetSection.Parent
            Target_index = SecParent.Child.index(TargetSection)
            NewPadSection = SectionNode(b'\x00\x00\x00\x19')
            SecParent.insertChild(NewPadSection, Target_index)
    elif type(TargetSection.Data.Header) == type(EFI_COMMON_SECTION_HEADER2()) and TargetSection.Data.Size < 16777215:
        New_Header = EFI_COMMON_SECTION_HEADER()
        New_Header.Type = TargetSection.Data.Header.Type
        New_Header.Size[0] = (TargetSection.Data.Size - 4) % 16 ** 2
        New_Header.Size[1] = (TargetSection.Data.Size - 4) % 16 ** 4 // 16 ** 2
        New_Header.Size[2] = (TargetSection.Data.Size - 4) // 16 ** 4
        TargetSection.Data.Header = New_Header
        TargetSection.Data.Size = TargetSection.Data.Header.SECTION_SIZE
        if TargetSection.LastRel.Data.IsPadSection:
            PadSectionModify(TargetSection.LastRel, -4)
        else:
            SecParent = TargetSection.Parent
            Target_index = SecParent.Child.index(TargetSection)
            NewPadSection = SectionNode(b'\x00\x00\x00\x19')
            SecParent.insertChild(NewPadSection, Target_index)

def ModifyFvExtData(TreeNode) -> None:
    if False:
        print('Hello World!')
    FvExtData = b''
    if TreeNode.Data.Header.ExtHeaderOffset:
        FvExtHeader = struct2stream(TreeNode.Data.ExtHeader)
        FvExtData += FvExtHeader
    if TreeNode.Data.Header.ExtHeaderOffset and TreeNode.Data.ExtEntryExist:
        FvExtEntry = struct2stream(TreeNode.Data.ExtEntry)
        FvExtData += FvExtEntry
    if FvExtData:
        InfoNode = TreeNode.Child[0]
        InfoNode.Data.Data = FvExtData + InfoNode.Data.Data[TreeNode.Data.ExtHeader.ExtHeaderSize:]
        InfoNode.Data.ModCheckSum()

def ModifyFvSystemGuid(TargetFv) -> None:
    if False:
        i = 10
        return i + 15
    if struct2stream(TargetFv.Data.Header.FileSystemGuid) == EFI_FIRMWARE_FILE_SYSTEM2_GUID_BYTE:
        TargetFv.Data.Header.FileSystemGuid = ModifyGuidFormat('5473C07A-3DCB-4dca-BD6F-1E9689E7349A')
    TargetFv.Data.ModCheckSum()
    TargetFv.Data.Data = b''
    for item in TargetFv.Child:
        if item.type == FFS_FREE_SPACE:
            TargetFv.Data.Data += item.Data.Data + item.Data.PadData
        else:
            TargetFv.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData

class FvHandler:

    def __init__(self, NewFfs, TargetFfs=None) -> None:
        if False:
            print('Hello World!')
        self.NewFfs = NewFfs
        self.TargetFfs = TargetFfs
        self.Status = False
        self.Remain_New_Free_Space = 0

    def CompressData(self, TargetTree) -> None:
        if False:
            i = 10
            return i + 15
        TreePath = TargetTree.GetTreePath()
        pos = len(TreePath)
        while pos:
            if not self.Status:
                if TreePath[pos - 1].type == SECTION_TREE and TreePath[pos - 1].Data.Type == 2:
                    self.CompressSectionData(TreePath[pos - 1], None, TreePath[pos - 1].Data.ExtHeader.SectionDefinitionGuid)
                elif pos == len(TreePath):
                    self.CompressSectionData(TreePath[pos - 1], pos)
                else:
                    self.CompressSectionData(TreePath[pos - 1], None)
            pos -= 1

    def CompressSectionData(self, TargetTree, pos: int, GuidTool=None) -> None:
        if False:
            print('Hello World!')
        NewData = b''
        temp_save_child = TargetTree.Child
        if TargetTree.Data:
            for item in temp_save_child:
                if item.type == SECTION_TREE and (not item.Data.OriData) and item.Data.ExtHeader:
                    NewData += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.Data + item.Data.PadData
                elif item.type == SECTION_TREE and item.Data.OriData and (not item.Data.ExtHeader):
                    NewData += struct2stream(item.Data.Header) + item.Data.OriData + item.Data.PadData
                elif item.type == SECTION_TREE and item.Data.OriData and item.Data.ExtHeader:
                    NewData += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.OriData + item.Data.PadData
                elif item.type == FFS_FREE_SPACE:
                    NewData += item.Data.Data + item.Data.PadData
                else:
                    NewData += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
            if TargetTree.type == FFS_TREE:
                New_Pad_Size = GetPadSize(len(NewData), 8)
                Size_delta = len(NewData) - len(TargetTree.Data.Data)
                ChangeSize(TargetTree, -Size_delta)
                Delta_Pad_Size = len(TargetTree.Data.PadData) - New_Pad_Size
                self.Remain_New_Free_Space += Delta_Pad_Size
                TargetTree.Data.PadData = b'\xff' * New_Pad_Size
                TargetTree.Data.ModCheckSum()
            elif TargetTree.type == FV_TREE or (TargetTree.type == SEC_FV_TREE and (not pos)):
                if self.Remain_New_Free_Space:
                    if TargetTree.Data.Free_Space:
                        TargetTree.Data.Free_Space += self.Remain_New_Free_Space
                        NewData += self.Remain_New_Free_Space * b'\xff'
                        TargetTree.Child[-1].Data.Data += self.Remain_New_Free_Space * b'\xff'
                    else:
                        TargetTree.Data.Data += self.Remain_New_Free_Space * b'\xff'
                        New_Free_Space = BIOSTREE('FREE_SPACE')
                        New_Free_Space.type = FFS_FREE_SPACE
                        New_Free_Space.Data = FreeSpaceNode(b'\xff' * self.Remain_New_Free_Space)
                        TargetTree.insertChild(New_Free_Space)
                    self.Remain_New_Free_Space = 0
                if TargetTree.type == SEC_FV_TREE:
                    Size_delta = len(NewData) + self.Remain_New_Free_Space - len(TargetTree.Data.Data)
                    TargetTree.Data.Header.FvLength += Size_delta
                TargetTree.Data.ModFvExt()
                TargetTree.Data.ModFvSize()
                TargetTree.Data.ModExtHeaderData()
                ModifyFvExtData(TargetTree)
                TargetTree.Data.ModCheckSum()
            elif TargetTree.type == SECTION_TREE and TargetTree.Data.Type != 2:
                New_Pad_Size = GetPadSize(len(NewData), 4)
                Size_delta = len(NewData) - len(TargetTree.Data.Data)
                ChangeSize(TargetTree, -Size_delta)
                if TargetTree.NextRel:
                    Delta_Pad_Size = len(TargetTree.Data.PadData) - New_Pad_Size
                    self.Remain_New_Free_Space += Delta_Pad_Size
                    TargetTree.Data.PadData = b'\x00' * New_Pad_Size
            TargetTree.Data.Data = NewData
        if GuidTool:
            guidtool = GUIDTools().__getitem__(struct2stream(GuidTool))
            if not guidtool.ifexist:
                logger.error('GuidTool {} is not found when decompressing {} file.\n'.format(guidtool.command, TargetTree.Parent.Data.Name))
                raise Exception('Process Failed: GuidTool not found!')
            CompressedData = guidtool.pack(TargetTree.Data.Data)
            if len(CompressedData) < len(TargetTree.Data.OriData):
                New_Pad_Size = GetPadSize(len(CompressedData), SECTION_COMMON_ALIGNMENT)
                Size_delta = len(CompressedData) - len(TargetTree.Data.OriData)
                ChangeSize(TargetTree, -Size_delta)
                if TargetTree.NextRel:
                    TargetTree.Data.PadData = b'\x00' * New_Pad_Size
                    self.Remain_New_Free_Space = len(TargetTree.Data.OriData) + len(TargetTree.Data.PadData) - len(CompressedData) - New_Pad_Size
                else:
                    TargetTree.Data.PadData = b''
                    self.Remain_New_Free_Space = len(TargetTree.Data.OriData) - len(CompressedData)
                TargetTree.Data.OriData = CompressedData
            elif len(CompressedData) == len(TargetTree.Data.OriData):
                TargetTree.Data.OriData = CompressedData
            elif len(CompressedData) > len(TargetTree.Data.OriData):
                New_Pad_Size = GetPadSize(len(CompressedData), SECTION_COMMON_ALIGNMENT)
                self.Remain_New_Free_Space = len(CompressedData) + New_Pad_Size - len(TargetTree.Data.OriData) - len(TargetTree.Data.PadData)
                self.ModifyTest(TargetTree, self.Remain_New_Free_Space)
                self.Status = True

    def ModifyTest(self, ParTree, Needed_Space: int) -> None:
        if False:
            while True:
                i = 10
        if Needed_Space > 0:
            if ParTree.type == FV_TREE or ParTree.type == SEC_FV_TREE:
                ParTree.Data.Data = b''
                Needed_Space = Needed_Space - ParTree.Data.Free_Space
                if Needed_Space < 0:
                    ParTree.Child[-1].Data.Data = b'\xff' * -Needed_Space
                    ParTree.Data.Free_Space = -Needed_Space
                    self.Status = True
                elif ParTree.type == FV_TREE:
                    self.Status = False
                else:
                    BlockSize = ParTree.Data.Header.BlockMap[0].Length
                    New_Add_Len = BlockSize - Needed_Space % BlockSize
                    if New_Add_Len % BlockSize:
                        ParTree.Child[-1].Data.Data = b'\xff' * New_Add_Len
                        ParTree.Data.Free_Space = New_Add_Len
                        Needed_Space += New_Add_Len
                    else:
                        ParTree.Child.remove(ParTree.Child[-1])
                        ParTree.Data.Free_Space = 0
                    ParTree.Data.Size += Needed_Space
                    ParTree.Data.Header.Fvlength = ParTree.Data.Size
                ModifyFvSystemGuid(ParTree)
                for item in ParTree.Child:
                    if item.type == FFS_FREE_SPACE:
                        ParTree.Data.Data += item.Data.Data + item.Data.PadData
                    else:
                        ParTree.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                ParTree.Data.ModFvExt()
                ParTree.Data.ModFvSize()
                ParTree.Data.ModExtHeaderData()
                ModifyFvExtData(ParTree)
                ParTree.Data.ModCheckSum()
            elif ParTree.type == FFS_TREE:
                ParTree.Data.Data = b''
                OriHeaderLen = ParTree.Data.HeaderLength
                for item in ParTree.Child:
                    if item.Data.OriData:
                        if item.Data.ExtHeader:
                            ParTree.Data.Data += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.OriData + item.Data.PadData
                        else:
                            ParTree.Data.Data += struct2stream(item.Data.Header) + item.Data.OriData + item.Data.PadData
                    elif item.Data.ExtHeader:
                        ParTree.Data.Data += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.Data + item.Data.PadData
                    else:
                        ParTree.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                ChangeSize(ParTree, -Needed_Space)
                ModifyFfsType(ParTree)
                Needed_Space += ParTree.Data.HeaderLength - OriHeaderLen
                New_Pad_Size = GetPadSize(ParTree.Data.Size, FFS_COMMON_ALIGNMENT)
                Delta_Pad_Size = New_Pad_Size - len(ParTree.Data.PadData)
                Needed_Space += Delta_Pad_Size
                ParTree.Data.PadData = b'\xff' * GetPadSize(ParTree.Data.Size, FFS_COMMON_ALIGNMENT)
                ParTree.Data.ModCheckSum()
            elif ParTree.type == SECTION_TREE:
                OriData = ParTree.Data.Data
                OriHeaderLen = ParTree.Data.HeaderLength
                ParTree.Data.Data = b''
                for item in ParTree.Child:
                    if item.type == SECTION_TREE and item.Data.ExtHeader and (item.Data.Type != 2):
                        ParTree.Data.Data += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.Data + item.Data.PadData
                    elif item.type == SECTION_TREE and item.Data.ExtHeader and (item.Data.Type == 2):
                        ParTree.Data.Data += struct2stream(item.Data.Header) + struct2stream(item.Data.ExtHeader) + item.Data.OriData + item.Data.PadData
                    else:
                        ParTree.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                if ParTree.Data.Type == 2:
                    guidtool = GUIDTools().__getitem__(struct2stream(ParTree.Data.ExtHeader.SectionDefinitionGuid))
                    if not guidtool.ifexist:
                        logger.error('GuidTool {} is not found when decompressing {} file.\n'.format(guidtool.command, ParTree.Parent.Data.Name))
                        raise Exception('Process Failed: GuidTool not found!')
                    CompressedData = guidtool.pack(ParTree.Data.Data)
                    Needed_Space = len(CompressedData) - len(ParTree.Data.OriData)
                    ParTree.Data.OriData = CompressedData
                    New_Size = ParTree.Data.HeaderLength + len(CompressedData)
                    ParTree.Data.Header.Size[0] = New_Size % 16 ** 2
                    ParTree.Data.Header.Size[1] = New_Size % 16 ** 4 // 16 ** 2
                    ParTree.Data.Header.Size[2] = New_Size // 16 ** 4
                    ParTree.Data.Size = ParTree.Data.Header.SECTION_SIZE
                    ModifySectionType(ParTree)
                    Needed_Space += ParTree.Data.HeaderLength - OriHeaderLen
                    if ParTree.NextRel:
                        New_Pad_Size = GetPadSize(ParTree.Data.Size, SECTION_COMMON_ALIGNMENT)
                        Delta_Pad_Size = New_Pad_Size - len(ParTree.Data.PadData)
                        ParTree.Data.PadData = b'\x00' * New_Pad_Size
                        Needed_Space += Delta_Pad_Size
                    else:
                        ParTree.Data.PadData = b''
                    if Needed_Space < 0:
                        self.Remain_New_Free_Space = len(ParTree.Data.OriData) - len(CompressedData)
                elif Needed_Space:
                    ChangeSize(ParTree, -Needed_Space)
                    ModifySectionType(ParTree)
                    Needed_Space += ParTree.Data.HeaderLength - OriHeaderLen
                    New_Pad_Size = GetPadSize(ParTree.Data.Size, SECTION_COMMON_ALIGNMENT)
                    Delta_Pad_Size = New_Pad_Size - len(ParTree.Data.PadData)
                    Needed_Space += Delta_Pad_Size
                    ParTree.Data.PadData = b'\x00' * New_Pad_Size
            NewParTree = ParTree.Parent
            ROOT_TYPE = [ROOT_FV_TREE, ROOT_FFS_TREE, ROOT_SECTION_TREE, ROOT_TREE]
            if NewParTree and NewParTree.type not in ROOT_TYPE:
                self.ModifyTest(NewParTree, Needed_Space)
        else:
            self.CompressData(ParTree)
            self.Status = True

    def ReplaceFfs(self) -> bool:
        if False:
            return 10
        logger.debug('Start Replacing Process......')
        TargetFv = self.TargetFfs.Parent
        if TargetFv.Data.Header.Attributes & EFI_FVB2_ERASE_POLARITY:
            self.NewFfs.Data.Header.State = c_uint8(~self.NewFfs.Data.Header.State)
        self.NewFfs.Data.PadData = b'\xff' * GetPadSize(self.NewFfs.Data.Size, FFS_COMMON_ALIGNMENT)
        if self.NewFfs.Data.Size >= self.TargetFfs.Data.Size:
            Needed_Space = self.NewFfs.Data.Size + len(self.NewFfs.Data.PadData) - self.TargetFfs.Data.Size - len(self.TargetFfs.Data.PadData)
            if TargetFv.Data.Free_Space >= Needed_Space:
                TargetFv.Child[-1].Data.Data = b'\xff' * (TargetFv.Data.Free_Space - Needed_Space)
                TargetFv.Data.Free_Space -= Needed_Space
                Target_index = TargetFv.Child.index(self.TargetFfs)
                TargetFv.Child.remove(self.TargetFfs)
                TargetFv.insertChild(self.NewFfs, Target_index)
                TargetFv.Data.ModFvExt()
                TargetFv.Data.ModFvSize()
                TargetFv.Data.ModExtHeaderData()
                ModifyFvExtData(TargetFv)
                TargetFv.Data.ModCheckSum()
                self.CompressData(TargetFv)
                self.Status = True
            elif TargetFv.type == FV_TREE:
                self.Status = False
            else:
                Needed_Space -= TargetFv.Data.Free_Space
                BlockSize = TargetFv.Data.Header.BlockMap[0].Length
                New_Add_Len = BlockSize - Needed_Space % BlockSize
                Target_index = TargetFv.Child.index(self.TargetFfs)
                if New_Add_Len % BlockSize:
                    TargetFv.Child[-1].Data.Data = b'\xff' * New_Add_Len
                    TargetFv.Data.Free_Space = New_Add_Len
                    Needed_Space += New_Add_Len
                    TargetFv.insertChild(self.NewFfs, Target_index)
                    TargetFv.Child.remove(self.TargetFfs)
                else:
                    TargetFv.Child.remove(self.TargetFfs)
                    TargetFv.Data.Free_Space = 0
                    TargetFv.insertChild(self.NewFfs)
                TargetFv.Data.Data = b''
                for item in TargetFv.Child:
                    if item.type == FFS_FREE_SPACE:
                        TargetFv.Data.Data += item.Data.Data + item.Data.PadData
                    else:
                        TargetFv.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                TargetFv.Data.Size += Needed_Space
                TargetFv.Data.Header.FvLength = TargetFv.Data.Size
                TargetFv.Data.ModFvExt()
                TargetFv.Data.ModFvSize()
                TargetFv.Data.ModExtHeaderData()
                ModifyFvExtData(TargetFv)
                TargetFv.Data.ModCheckSum()
                self.ModifyTest(TargetFv.Parent, Needed_Space)
        else:
            New_Free_Space = self.TargetFfs.Data.Size - self.NewFfs.Data.Size
            if TargetFv.Data.Free_Space:
                TargetFv.Child[-1].Data.Data += b'\xff' * New_Free_Space
                TargetFv.Data.Free_Space += New_Free_Space
                Target_index = TargetFv.Child.index(self.TargetFfs)
                TargetFv.Child.remove(self.TargetFfs)
                TargetFv.insertChild(self.NewFfs, Target_index)
                self.Status = True
            else:
                New_Free_Space_Tree = BIOSTREE('FREE_SPACE')
                New_Free_Space_Tree.type = FFS_FREE_SPACE
                New_Free_Space_Tree.Data = FfsNode(b'\xff' * New_Free_Space)
                TargetFv.Data.Free_Space = New_Free_Space
                TargetFv.insertChild(New_Free_Space)
                Target_index = TargetFv.Child.index(self.TargetFfs)
                TargetFv.Child.remove(self.TargetFfs)
                TargetFv.insertChild(self.NewFfs, Target_index)
                self.Status = True
            TargetFv.Data.ModFvExt()
            TargetFv.Data.ModFvSize()
            TargetFv.Data.ModExtHeaderData()
            ModifyFvExtData(TargetFv)
            TargetFv.Data.ModCheckSum()
            self.CompressData(TargetFv)
        logger.debug('Done!')
        return self.Status

    def AddFfs(self) -> bool:
        if False:
            i = 10
            return i + 15
        logger.debug('Start Adding Process......')
        self.NewFfs.Data.PadData = b'\xff' * GetPadSize(self.NewFfs.Data.Size, FFS_COMMON_ALIGNMENT)
        if self.TargetFfs.type == FFS_FREE_SPACE:
            TargetLen = self.NewFfs.Data.Size + len(self.NewFfs.Data.PadData) - self.TargetFfs.Data.Size - len(self.TargetFfs.Data.PadData)
            TargetFv = self.TargetFfs.Parent
            if TargetFv.Data.Header.Attributes & EFI_FVB2_ERASE_POLARITY:
                self.NewFfs.Data.Header.State = c_uint8(~self.NewFfs.Data.Header.State)
            if TargetLen < 0:
                self.TargetFfs.Data.Data = b'\xff' * -TargetLen
                TargetFv.Data.Free_Space = -TargetLen
                TargetFv.Data.ModFvExt()
                TargetFv.Data.ModExtHeaderData()
                ModifyFvExtData(TargetFv)
                TargetFv.Data.ModCheckSum()
                TargetFv.insertChild(self.NewFfs, -1)
                ModifyFfsType(self.NewFfs)
                self.CompressData(TargetFv)
                self.Status = True
            elif TargetLen == 0:
                TargetFv.Child.remove(self.TargetFfs)
                TargetFv.insertChild(self.NewFfs)
                ModifyFfsType(self.NewFfs)
                self.CompressData(TargetFv)
                self.Status = True
            elif TargetFv.type == FV_TREE:
                self.Status = False
            elif TargetFv.type == SEC_FV_TREE:
                BlockSize = TargetFv.Data.Header.BlockMap[0].Length
                New_Add_Len = BlockSize - TargetLen % BlockSize
                if New_Add_Len % BlockSize:
                    self.TargetFfs.Data.Data = b'\xff' * New_Add_Len
                    self.TargetFfs.Data.Size = New_Add_Len
                    TargetLen += New_Add_Len
                    TargetFv.insertChild(self.NewFfs, -1)
                    TargetFv.Data.Free_Space = New_Add_Len
                else:
                    TargetFv.Child.remove(self.TargetFfs)
                    TargetFv.insertChild(self.NewFfs)
                    TargetFv.Data.Free_Space = 0
                ModifyFfsType(self.NewFfs)
                ModifyFvSystemGuid(TargetFv)
                TargetFv.Data.Data = b''
                for item in TargetFv.Child:
                    if item.type == FFS_FREE_SPACE:
                        TargetFv.Data.Data += item.Data.Data + item.Data.PadData
                    else:
                        TargetFv.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                TargetFv.Data.Size += TargetLen
                TargetFv.Data.Header.FvLength = TargetFv.Data.Size
                TargetFv.Data.ModFvExt()
                TargetFv.Data.ModFvSize()
                TargetFv.Data.ModExtHeaderData()
                ModifyFvExtData(TargetFv)
                TargetFv.Data.ModCheckSum()
                self.ModifyTest(TargetFv.Parent, TargetLen)
        else:
            TargetLen = self.NewFfs.Data.Size + len(self.NewFfs.Data.PadData)
            TargetFv = self.TargetFfs.Parent
            if TargetFv.Data.Header.Attributes & EFI_FVB2_ERASE_POLARITY:
                self.NewFfs.Data.Header.State = c_uint8(~self.NewFfs.Data.Header.State)
            if TargetFv.type == FV_TREE:
                self.Status = False
            elif TargetFv.type == SEC_FV_TREE:
                BlockSize = TargetFv.Data.Header.BlockMap[0].Length
                New_Add_Len = BlockSize - TargetLen % BlockSize
                if New_Add_Len % BlockSize:
                    New_Free_Space = BIOSTREE('FREE_SPACE')
                    New_Free_Space.type = FFS_FREE_SPACE
                    New_Free_Space.Data = FreeSpaceNode(b'\xff' * New_Add_Len)
                    TargetLen += New_Add_Len
                    TargetFv.Data.Free_Space = New_Add_Len
                    TargetFv.insertChild(self.NewFfs)
                    TargetFv.insertChild(New_Free_Space)
                else:
                    TargetFv.insertChild(self.NewFfs)
                ModifyFfsType(self.NewFfs)
                ModifyFvSystemGuid(TargetFv)
                TargetFv.Data.Data = b''
                for item in TargetFv.Child:
                    if item.type == FFS_FREE_SPACE:
                        TargetFv.Data.Data += item.Data.Data + item.Data.PadData
                    else:
                        TargetFv.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
                TargetFv.Data.Size += TargetLen
                TargetFv.Data.Header.FvLength = TargetFv.Data.Size
                TargetFv.Data.ModFvExt()
                TargetFv.Data.ModFvSize()
                TargetFv.Data.ModExtHeaderData()
                ModifyFvExtData(TargetFv)
                TargetFv.Data.ModCheckSum()
                self.ModifyTest(TargetFv.Parent, TargetLen)
        logger.debug('Done!')
        return self.Status

    def DeleteFfs(self) -> bool:
        if False:
            while True:
                i = 10
        logger.debug('Start Deleting Process......')
        Delete_Ffs = self.TargetFfs
        Delete_Fv = Delete_Ffs.Parent
        Add_Free_Space = Delete_Ffs.Data.Size + len(Delete_Ffs.Data.PadData)
        if Delete_Fv.Data.Free_Space:
            if Delete_Fv.type == SEC_FV_TREE:
                Used_Size = Delete_Fv.Data.Size - Delete_Fv.Data.Free_Space - Add_Free_Space
                BlockSize = Delete_Fv.Data.Header.BlockMap[0].Length
                New_Free_Space = BlockSize - Used_Size % BlockSize
                self.Remain_New_Free_Space += Delete_Fv.Data.Free_Space + Add_Free_Space - New_Free_Space
                Delete_Fv.Child[-1].Data.Data = New_Free_Space * b'\xff'
                Delete_Fv.Data.Free_Space = New_Free_Space
            else:
                Used_Size = Delete_Fv.Data.Size - Delete_Fv.Data.Free_Space - Add_Free_Space
                Delete_Fv.Child[-1].Data.Data += Add_Free_Space * b'\xff'
                Delete_Fv.Data.Free_Space += Add_Free_Space
                New_Free_Space = Delete_Fv.Data.Free_Space
        else:
            if Delete_Fv.type == SEC_FV_TREE:
                Used_Size = Delete_Fv.Data.Size - Add_Free_Space
                BlockSize = Delete_Fv.Data.Header.BlockMap[0].Length
                New_Free_Space = BlockSize - Used_Size % BlockSize
                self.Remain_New_Free_Space += Add_Free_Space - New_Free_Space
                Add_Free_Space = New_Free_Space
            else:
                Used_Size = Delete_Fv.Data.Size - Add_Free_Space
                New_Free_Space = Add_Free_Space
            New_Free_Space_Info = FfsNode(Add_Free_Space * b'\xff')
            New_Free_Space_Info.Data = Add_Free_Space * b'\xff'
            New_Ffs_Tree = BIOSTREE(New_Free_Space_Info.Name)
            New_Ffs_Tree.type = FFS_FREE_SPACE
            New_Ffs_Tree.Data = New_Free_Space_Info
            Delete_Fv.insertChild(New_Ffs_Tree)
            Delete_Fv.Data.Free_Space = Add_Free_Space
        Delete_Fv.Child.remove(Delete_Ffs)
        Delete_Fv.Data.Header.FvLength = Used_Size + New_Free_Space
        Delete_Fv.Data.ModFvExt()
        Delete_Fv.Data.ModFvSize()
        Delete_Fv.Data.ModExtHeaderData()
        ModifyFvExtData(Delete_Fv)
        Delete_Fv.Data.ModCheckSum()
        self.CompressData(Delete_Fv)
        self.Status = True
        logger.debug('Done!')
        return self.Status

    def ShrinkFv(self) -> bool:
        if False:
            print('Hello World!')
        TargetFv = self.NewFfs
        TargetFv.Data.Data = b''
        if not TargetFv.Data.Free_Space:
            self.Status = True
        else:
            BlockSize = TargetFv.Data.Header.BlockMap[0].Length
            New_Free_Space = TargetFv.Data.Free_Space % BlockSize
            Removed_Space = TargetFv.Data.Free_Space - New_Free_Space
            TargetFv.Child[-1].Data.Data = b'\xff' * New_Free_Space
            TargetFv.Data.Size -= Removed_Space
            TargetFv.Data.Header.Fvlength = TargetFv.Data.Size
            ModifyFvSystemGuid(TargetFv)
            for item in TargetFv.Child:
                if item.type == FFS_FREE_SPACE:
                    TargetFv.Data.Data += item.Data.Data + item.Data.PadData
                else:
                    TargetFv.Data.Data += struct2stream(item.Data.Header) + item.Data.Data + item.Data.PadData
            TargetFv.Data.ModFvExt()
            TargetFv.Data.ModFvSize()
            TargetFv.Data.ModExtHeaderData()
            ModifyFvExtData(TargetFv)
            TargetFv.Data.ModCheckSum()
            self.Status = True
        return self.Status