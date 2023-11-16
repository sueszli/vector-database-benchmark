import pythoncom
from win32com.server.policy import DesignatedWrapPolicy
from win32com.shell import shell, shellcon
tsf_flags = [(k, v) for (k, v) in list(shellcon.__dict__.items()) if k.startswith('TSF_')]

def decode_flags(flags):
    if False:
        print('Hello World!')
    if flags == 0:
        return 'TSF_NORMAL'
    flag_txt = ''
    for (k, v) in tsf_flags:
        if flags & v:
            if flag_txt:
                flag_txt = flag_txt + '|' + k
            else:
                flag_txt = k
    return flag_txt

class FileOperationProgressSink(DesignatedWrapPolicy):
    _com_interfaces_ = [shell.IID_IFileOperationProgressSink]
    _public_methods_ = ['StartOperations', 'FinishOperations', 'PreRenameItem', 'PostRenameItem', 'PreMoveItem', 'PostMoveItem', 'PreCopyItem', 'PostCopyItem', 'PreDeleteItem', 'PostDeleteItem', 'PreNewItem', 'PostNewItem', 'UpdateProgress', 'ResetTimer', 'PauseTimer', 'ResumeTimer']

    def __init__(self):
        if False:
            while True:
                i = 10
        self._wrap_(self)

    def StartOperations(self):
        if False:
            return 10
        print('StartOperations')

    def FinishOperations(self, Result):
        if False:
            for i in range(10):
                print('nop')
        print('FinishOperations: HRESULT ', Result)

    def PreRenameItem(self, Flags, Item, NewName):
        if False:
            return 10
        print('PreRenameItem: Renaming ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + NewName)

    def PostRenameItem(self, Flags, Item, NewName, hrRename, NewlyCreated):
        if False:
            return 10
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = 'not renamed, HRESULT ' + str(hrRename)
        print('PostRenameItem: renamed ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + newfile)

    def PreMoveItem(self, Flags, Item, DestinationFolder, NewName):
        if False:
            for i in range(10):
                print('nop')
        print('PreMoveItem: Moving ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING) + '\\' + str(NewName))

    def PostMoveItem(self, Flags, Item, DestinationFolder, NewName, hrMove, NewlyCreated):
        if False:
            return 10
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = 'not copied, HRESULT ' + str(hrMove)
        print('PostMoveItem: Moved ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + newfile)

    def PreCopyItem(self, Flags, Item, DestinationFolder, NewName):
        if False:
            while True:
                i = 10
        if not NewName:
            NewName = ''
        print('PreCopyItem: Copying ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING) + '\\' + NewName)
        print('Flags: ', decode_flags(Flags))

    def PostCopyItem(self, Flags, Item, DestinationFolder, NewName, hrCopy, NewlyCreated):
        if False:
            for i in range(10):
                print('nop')
        if NewlyCreated is not None:
            newfile = NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING)
        else:
            newfile = 'not copied, HRESULT ' + str(hrCopy)
        print('PostCopyItem: Copied ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING) + ' to ' + newfile)
        print('Flags: ', decode_flags(Flags))

    def PreDeleteItem(self, Flags, Item):
        if False:
            i = 10
            return i + 15
        print('PreDeleteItem: Deleting ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING))

    def PostDeleteItem(self, Flags, Item, hrDelete, NewlyCreated):
        if False:
            print('Hello World!')
        print('PostDeleteItem: Deleted ' + Item.GetDisplayName(shellcon.SHGDN_FORPARSING))
        if NewlyCreated:
            print('\tMoved to recycle bin - ' + NewlyCreated.GetDisplayName(shellcon.SHGDN_FORPARSING))

    def PreNewItem(self, Flags, DestinationFolder, NewName):
        if False:
            i = 10
            return i + 15
        print('PreNewItem: Creating ' + DestinationFolder.GetDisplayName(shellcon.SHGDN_FORPARSING) + '\\' + NewName)

    def PostNewItem(self, Flags, DestinationFolder, NewName, TemplateName, FileAttributes, hrNew, NewItem):
        if False:
            return 10
        print('PostNewItem: Created ' + NewItem.GetDisplayName(shellcon.SHGDN_FORPARSING))

    def UpdateProgress(self, WorkTotal, WorkSoFar):
        if False:
            for i in range(10):
                print('nop')
        print('UpdateProgress: ', WorkSoFar, WorkTotal)

    def ResetTimer(self):
        if False:
            for i in range(10):
                print('nop')
        print('ResetTimer')

    def PauseTimer(self):
        if False:
            for i in range(10):
                print('nop')
        print('PauseTimer')

    def ResumeTimer(self):
        if False:
            for i in range(10):
                print('nop')
        print('ResumeTimer')

def CreateSink():
    if False:
        for i in range(10):
            print('nop')
    return pythoncom.WrapObject(FileOperationProgressSink(), shell.IID_IFileOperationProgressSink)