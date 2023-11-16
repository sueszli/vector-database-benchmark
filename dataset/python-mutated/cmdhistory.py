import volatility.obj as obj
import volatility.plugins.common as common
import volatility.utils as utils
import volatility.win32.tasks as tasks
import volatility.debug as debug
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address, Hex
MAX_HISTORY_DEFAULT = 50
conhost_types_x86 = {'_COMMAND': [None, {'CmdLength': [0, ['unsigned short']], 'Cmd': [2, ['String', dict(encoding='utf16', length=lambda x: x.CmdLength)]]}], '_COMMAND_HISTORY': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'Flags': [8, ['Flags', {'bitmap': {'Allocated': 0, 'Reset': 1}}]], 'Application': [12, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'CommandCount': [16, ['short']], 'LastAdded': [18, ['short']], 'LastDisplayed': [20, ['short']], 'FirstCommand': [22, ['short']], 'CommandCountMax': [24, ['short']], 'ProcessHandle': [28, ['unsigned int']], 'PopupList': [32, ['_LIST_ENTRY']], 'CommandBucket': [40, ['array', lambda x: x.CommandCount, ['pointer', ['_COMMAND']]]]}], '_ALIAS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'SourceLength': [8, ['unsigned short']], 'TargetLength': [10, ['unsigned short']], 'Source': [12, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.SourceLength)]]], 'Target': [16, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.TargetLength)]]]}], '_EXE_ALIAS_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ExeLength': [8, ['unsigned short']], 'ExeName': [12, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.ExeLength * 2)]]], 'AliasList': [16, ['_LIST_ENTRY']]}], '_POPUP_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']]}], '_CONSOLE_INFORMATION': [None, {'CurrentScreenBuffer': [152, ['pointer', ['_SCREEN_INFORMATION']]], 'ScreenBuffer': [156, ['pointer', ['_SCREEN_INFORMATION']]], 'HistoryList': [212, ['_LIST_ENTRY']], 'ProcessList': [24, ['_LIST_ENTRY']], 'ExeAliasList': [220, ['_LIST_ENTRY']], 'HistoryBufferCount': [228, ['unsigned short']], 'HistoryBufferMax': [230, ['unsigned short']], 'CommandHistorySize': [232, ['unsigned short']], 'OriginalTitle': [236, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Title': [240, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CONSOLE_PROCESS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ProcessHandle': [8, ['unsigned int']]}], '_SCREEN_INFORMATION': [None, {'ScreenX': [8, ['short']], 'ScreenY': [10, ['short']], 'Rows': [60, ['pointer', ['array', lambda x: x.ScreenY, ['_ROW']]]], 'Next': [220, ['pointer', ['_SCREEN_INFORMATION']]]}], '_ROW': [28, {'Chars': [8, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}]}
conhost_types_x64 = {'_COMMAND': [None, {'CmdLength': [0, ['unsigned short']], 'Cmd': [2, ['String', dict(encoding='utf16', length=lambda x: x.CmdLength)]]}], '_COMMAND_HISTORY': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'Flags': [16, ['Flags', {'bitmap': {'Allocated': 0, 'Reset': 1}}]], 'Application': [24, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'CommandCount': [32, ['short']], 'LastAdded': [34, ['short']], 'LastDisplayed': [36, ['short']], 'FirstCommand': [38, ['short']], 'CommandCountMax': [40, ['short']], 'ProcessHandle': [48, ['address']], 'PopupList': [56, ['_LIST_ENTRY']], 'CommandBucket': [72, ['array', lambda x: x.CommandCount, ['pointer', ['_COMMAND']]]]}], '_ALIAS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'SourceLength': [16, ['unsigned short']], 'TargetLength': [18, ['unsigned short']], 'Source': [24, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.SourceLength)]]], 'Target': [32, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.TargetLength)]]]}], '_EXE_ALIAS_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ExeLength': [16, ['unsigned short']], 'ExeName': [24, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.ExeLength * 2)]]], 'AliasList': [32, ['_LIST_ENTRY']]}], '_POPUP_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']]}], '_CONSOLE_INFORMATION': [None, {'ProcessList': [40, ['_LIST_ENTRY']], 'CurrentScreenBuffer': [224, ['pointer', ['_SCREEN_INFORMATION']]], 'ScreenBuffer': [232, ['pointer', ['_SCREEN_INFORMATION']]], 'HistoryList': [328, ['_LIST_ENTRY']], 'ExeAliasList': [344, ['_LIST_ENTRY']], 'HistoryBufferCount': [360, ['unsigned short']], 'HistoryBufferMax': [362, ['unsigned short']], 'CommandHistorySize': [364, ['unsigned short']], 'OriginalTitle': [368, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Title': [376, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CONSOLE_PROCESS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ProcessHandle': [16, ['unsigned int']]}], '_SCREEN_INFORMATION': [None, {'ScreenX': [8, ['short']], 'ScreenY': [10, ['short']], 'Rows': [72, ['pointer', ['array', lambda x: x.ScreenY, ['_ROW']]]], 'Next': [296, ['pointer', ['_SCREEN_INFORMATION']]]}], '_ROW': [40, {'Chars': [8, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}]}
winsrv_types_x86 = {'_COMMAND': [None, {'CmdLength': [0, ['unsigned short']], 'Cmd': [2, ['String', dict(encoding='utf16', length=lambda x: x.CmdLength)]]}], '_COMMAND_HISTORY': [None, {'Flags': [0, ['Flags', {'bitmap': {'Allocated': 0, 'Reset': 1}}]], 'ListEntry': [4, ['_LIST_ENTRY']], 'Application': [12, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'CommandCount': [16, ['short']], 'LastAdded': [18, ['short']], 'LastDisplayed': [20, ['short']], 'FirstCommand': [22, ['short']], 'CommandCountMax': [24, ['short']], 'ProcessHandle': [28, ['unsigned int']], 'PopupList': [32, ['_LIST_ENTRY']], 'CommandBucket': [40, ['array', lambda x: x.CommandCount, ['pointer', ['_COMMAND']]]]}], '_ALIAS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'SourceLength': [8, ['unsigned short']], 'TargetLength': [10, ['unsigned short']], 'Source': [12, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.SourceLength)]]], 'Target': [16, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.TargetLength)]]]}], '_EXE_ALIAS_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ExeLength': [8, ['unsigned short']], 'ExeName': [12, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.ExeLength * 2)]]], 'AliasList': [16, ['_LIST_ENTRY']]}], '_POPUP_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']]}], '_CONSOLE_INFORMATION': [None, {'CurrentScreenBuffer': [176, ['pointer', ['_SCREEN_INFORMATION']]], 'ScreenBuffer': [180, ['pointer', ['_SCREEN_INFORMATION']]], 'HistoryList': [264, ['_LIST_ENTRY']], 'ProcessList': [256, ['_LIST_ENTRY']], 'ExeAliasList': [272, ['_LIST_ENTRY']], 'HistoryBufferCount': [280, ['unsigned short']], 'HistoryBufferMax': [282, ['unsigned short']], 'CommandHistorySize': [284, ['unsigned short']], 'OriginalTitle': [292, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Title': [296, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CONSOLE_PROCESS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ProcessHandle': [8, ['unsigned int']], 'Process': [12, ['pointer', ['_CSR_PROCESS']]]}], '_SCREEN_INFORMATION': [None, {'Console': [0, ['pointer', ['_CONSOLE_INFORMATION']]], 'ScreenX': [36, ['short']], 'ScreenY': [38, ['short']], 'Rows': [88, ['pointer', ['array', lambda x: x.ScreenY, ['_ROW']]]], 'Next': [248, ['pointer', ['_SCREEN_INFORMATION']]]}], '_ROW': [28, {'Chars': [8, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CSR_PROCESS': [96, {'ClientId': [0, ['_CLIENT_ID']], 'ListLink': [8, ['_LIST_ENTRY']], 'ThreadList': [16, ['_LIST_ENTRY']], 'NtSession': [24, ['pointer', ['_CSR_NT_SESSION']]], 'ClientPort': [28, ['pointer', ['void']]], 'ClientViewBase': [32, ['pointer', ['unsigned char']]], 'ClientViewBounds': [36, ['pointer', ['unsigned char']]], 'ProcessHandle': [40, ['pointer', ['void']]], 'SequenceNumber': [44, ['unsigned long']], 'Flags': [48, ['unsigned long']], 'DebugFlags': [52, ['unsigned long']], 'ReferenceCount': [56, ['unsigned long']], 'ProcessGroupId': [60, ['unsigned long']], 'ProcessGroupSequence': [64, ['unsigned long']], 'LastMessageSequence': [68, ['unsigned long']], 'NumOutstandingMessages': [72, ['unsigned long']], 'ShutdownLevel': [76, ['unsigned long']], 'ShutdownFlags': [80, ['unsigned long']], 'Luid': [84, ['_LUID']], 'ServerDllPerProcessData': [92, ['array', 1, ['pointer', ['void']]]]}]}
winsrv_types_x64 = {'_COMMAND': [None, {'CmdLength': [0, ['unsigned short']], 'Cmd': [2, ['String', dict(encoding='utf16', length=lambda x: x.CmdLength)]]}], '_COMMAND_HISTORY': [None, {'Flags': [0, ['Flags', {'bitmap': {'Allocated': 0, 'Reset': 1}}]], 'ListEntry': [8, ['_LIST_ENTRY']], 'Application': [24, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'CommandCount': [32, ['short']], 'LastAdded': [34, ['short']], 'LastDisplayed': [36, ['short']], 'FirstCommand': [38, ['short']], 'CommandCountMax': [40, ['short']], 'ProcessHandle': [48, ['unsigned int']], 'PopupList': [56, ['_LIST_ENTRY']], 'CommandBucket': [72, ['array', lambda x: x.CommandCount, ['pointer', ['_COMMAND']]]]}], '_ALIAS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'SourceLength': [16, ['unsigned short']], 'TargetLength': [18, ['unsigned short']], 'Source': [20, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.SourceLength)]]], 'Target': [28, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.TargetLength)]]]}], '_EXE_ALIAS_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ExeLength': [16, ['unsigned short']], 'ExeName': [18, ['pointer', ['String', dict(encoding='utf16', length=lambda x: x.ExeLength * 2)]]], 'AliasList': [26, ['_LIST_ENTRY']]}], '_POPUP_LIST': [None, {'ListEntry': [0, ['_LIST_ENTRY']]}], '_CONSOLE_INFORMATION': [None, {'CurrentScreenBuffer': [232, ['pointer', ['_SCREEN_INFORMATION']]], 'ScreenBuffer': [240, ['pointer', ['_SCREEN_INFORMATION']]], 'HistoryList': [392, ['_LIST_ENTRY']], 'ProcessList': [376, ['_LIST_ENTRY']], 'ExeAliasList': [408, ['_LIST_ENTRY']], 'HistoryBufferCount': [424, ['unsigned short']], 'HistoryBufferMax': [426, ['unsigned short']], 'CommandHistorySize': [428, ['unsigned short']], 'OriginalTitle': [432, ['pointer', ['String', dict(encoding='utf16', length=256)]]], 'Title': [440, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CONSOLE_PROCESS': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'ProcessHandle': [16, ['unsigned int']], 'Process': [24, ['pointer', ['_CSR_PROCESS']]]}], '_SCREEN_INFORMATION': [None, {'Console': [0, ['pointer', ['_CONSOLE_INFORMATION']]], 'ScreenX': [40, ['short']], 'ScreenY': [42, ['short']], 'Rows': [104, ['pointer', ['array', lambda x: x.ScreenY, ['_ROW']]]], 'Next': [296, ['pointer', ['_SCREEN_INFORMATION']]]}], '_ROW': [40, {'Chars': [8, ['pointer', ['String', dict(encoding='utf16', length=256)]]]}], '_CSR_PROCESS': [96, {'ClientId': [0, ['_CLIENT_ID']], 'ListLink': [8, ['_LIST_ENTRY']], 'ThreadList': [16, ['_LIST_ENTRY']], 'NtSession': [24, ['pointer', ['_CSR_NT_SESSION']]], 'ClientPort': [28, ['pointer', ['void']]], 'ClientViewBase': [32, ['pointer', ['unsigned char']]], 'ClientViewBounds': [36, ['pointer', ['unsigned char']]], 'ProcessHandle': [40, ['pointer', ['void']]], 'SequenceNumber': [44, ['unsigned long']], 'Flags': [48, ['unsigned long']], 'DebugFlags': [52, ['unsigned long']], 'ReferenceCount': [56, ['unsigned long']], 'ProcessGroupId': [60, ['unsigned long']], 'ProcessGroupSequence': [64, ['unsigned long']], 'LastMessageSequence': [68, ['unsigned long']], 'NumOutstandingMessages': [72, ['unsigned long']], 'ShutdownLevel': [76, ['unsigned long']], 'ShutdownFlags': [80, ['unsigned long']], 'Luid': [84, ['_LUID']], 'ServerDllPerProcessData': [92, ['array', 1, ['pointer', ['void']]]]}]}

class _CONSOLE_INFORMATION(obj.CType):
    """ object class for console information structs """

    def get_histories(self):
        if False:
            i = 10
            return i + 15
        for hist in self.HistoryList.list_of_type('_COMMAND_HISTORY', 'ListEntry'):
            yield hist

    def get_exe_aliases(self):
        if False:
            return 10
        'Generator for exe aliases.\n\n        There is one _EXE_ALIAS_LIST for each executable \n        (i.e. C:\\windows\\system32\\cmd.exe) with registered\n        aliases. The _EXE_ALIAS_LIST.AliasList contains \n        one _ALIAS structure for each specific mapping.\n\n        See GetConsoleAliasExes, GetConsoleAliases, and  \n        AddConsoleAlias. \n        '
        for exe_alias in self.ExeAliasList.list_of_type('_EXE_ALIAS_LIST', 'ListEntry'):
            yield exe_alias

    def get_processes(self):
        if False:
            while True:
                i = 10
        "Generator for processes attached to the console. \n\n        Multiple processes can be attached to the same\n        console (usually as a result of inheritance from a \n        parent process or by duplicating another process's \n        console handle). Internally, they are tracked as \n        _CONSOLE_PROCESS structures in this linked list. \n\n        See GetConsoleProcessList and AttachConsole. \n        "
        for h in self.ProcessList.list_of_type('_CONSOLE_PROCESS', 'ListEntry'):
            yield h

    def get_screens(self):
        if False:
            while True:
                i = 10
        'Generator for screens in the console. \n\n        A console can have multiple screen buffers at a time, \n        but only the current/active one is displayed. \n\n        Multiple screens are tracked using the singly-linked\n        list _SCREEN_INFORMATION.Next. \n    \n        See CreateConsoleScreenBuffer \n        '
        screens = [self.CurrentScreenBuffer]
        if self.ScreenBuffer not in screens:
            screens.append(self.ScreenBuffer)
        for screen in screens:
            cur = screen
            while cur and cur.v() != 0:
                yield cur
                cur = cur.Next.dereference()

class _CONSOLE_PROCESS(obj.CType):
    """ object class for console process """

    def reference_object_by_handle(self):
        if False:
            while True:
                i = 10
        ' Given a process handle, return a reference to \n        the _EPROCESS object. This function is similar to \n        the kernel API ObReferenceObjectByHandle. '
        console_information = self.obj_parent
        parent_process = console_information.obj_parent
        for h in parent_process.ObjectTable.handles():
            if h.HandleValue == self.ProcessHandle:
                return h.dereference_as('_EPROCESS')
        return obj.NoneObject('Could not find process in handle table')

class _SCREEN_INFORMATION(obj.CType):
    """ object class for screen information """

    def get_buffer(self, truncate=True):
        if False:
            print('Hello World!')
        "Get the screen buffer. \n\n        The screen buffer is comprised of the screen's Y \n        coordinate which tells us the number of rows and \n        the X coordinate which tells us the width of each\n        row in characters. These together provide all of \n        the input and output that users see when the \n        console is displayed. \n\n        @param truncate: True if the empty rows at the \n        end (i.e. bottom) of the screen buffer should be \n        supressed.\n        "
        rows = []
        for (_, row) in enumerate(self.Rows.dereference()):
            if row.Chars.is_valid():
                rows.append(str(row.Chars.dereference())[0:self.ScreenX])
        if truncate:
            non_empty_index = 0
            for (index, row) in enumerate(reversed(rows)):
                if row.count(' ') != min(self.ScreenX, 128):
                    non_empty_index = index
                    break
            if non_empty_index == 0:
                rows = []
            else:
                rows = rows[0:len(rows) - non_empty_index]
        return rows

class _EXE_ALIAS_LIST(obj.CType):
    """ object class for alias lists """

    def get_aliases(self):
        if False:
            return 10
        'Generator for the individual aliases for a\n        particular executable.'
        for alias in self.AliasList.list_of_type('_ALIAS', 'ListEntry'):
            yield alias

class _COMMAND_HISTORY(obj.CType):
    """ object class for command histories """

    def is_valid(self, max_history=MAX_HISTORY_DEFAULT):
        if False:
            i = 10
            return i + 15
        'Override BaseObject.is_valid with some additional\n        checks specific to _COMMAND_HISTORY objects.'
        if not obj.CType.is_valid(self):
            return False
        if self.CommandCount < 0 or self.CommandCount > max_history:
            return False
        if self.LastAdded < -1 or self.LastAdded > max_history:
            return False
        if self.LastDisplayed < -1 or self.LastDisplayed > max_history:
            return False
        if self.FirstCommand < 0 or self.FirstCommand > max_history:
            return False
        if self.FirstCommand != 0 and self.FirstCommand != self.LastAdded + 1:
            return False
        if self.ProcessHandle <= 0 or self.ProcessHandle > 65535:
            return False
        Popup = obj.Object('_POPUP_LIST', offset=self.PopupList.Flink, vm=self.obj_vm)
        if Popup.ListEntry.Blink != self.PopupList.obj_offset:
            return False
        return True

    def get_commands(self):
        if False:
            while True:
                i = 10
        'Generator for commands in the history buffer. \n\n        The CommandBucket is an array of pointers to _COMMAND \n        structures. The array size is CommandCount. Once CommandCount \n        is reached, the oldest commands are cycled out and the \n        rest are coalesced. \n        '
        for (i, cmd) in enumerate(self.CommandBucket):
            if cmd:
                yield (i, cmd.dereference())

class CmdHistoryVTypesx86(obj.ProfileModification):
    """This modification applies the vtypes for 32bit 
    Windows up to Windows 7."""
    before = ['WindowsObjectClasses']

    def check(self, profile):
        if False:
            return 10
        m = profile.metadata
        return m.get('os', None) == 'windows' and m.get('memory_model', '32bit') == '32bit' and (m.get('major') < 6 or (m.get('major') == 6 and m.get('minor') < 1))

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(winsrv_types_x86)

class CmdHistoryVTypesx64(obj.ProfileModification):
    """This modification applies the vtypes for 64bit 
    Windows up to Windows 7."""
    before = ['WindowsObjectClasses']

    def check(self, profile):
        if False:
            for i in range(10):
                print('nop')
        m = profile.metadata
        return m.get('os', None) == 'windows' and m.get('memory_model', '32bit') == '64bit' and (m.get('major') < 6 or (m.get('major') == 6 and m.get('minor') < 1))

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update(winsrv_types_x64)

class CmdHistoryVTypesWin7x86(obj.ProfileModification):
    """This modification applies the vtypes for 32bit 
    Windows starting with Windows 7."""
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x >= 1, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(conhost_types_x86)

class CmdHistoryVTypesWin7x64(obj.ProfileModification):
    """This modification applies the vtypes for 64bit 
    Windows starting with Windows 7."""
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 6, 'minor': lambda x: x >= 1, 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update(conhost_types_x64)

class CmdHistoryObjectClasses(obj.ProfileModification):
    """This modification applies the object classes for all 
    versions of 32bit Windows."""
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            return 10
        profile.object_classes.update({'_CONSOLE_INFORMATION': _CONSOLE_INFORMATION, '_SCREEN_INFORMATION': _SCREEN_INFORMATION, '_EXE_ALIAS_LIST': _EXE_ALIAS_LIST, '_COMMAND_HISTORY': _COMMAND_HISTORY, '_CONSOLE_PROCESS': _CONSOLE_PROCESS})

class CmdScan(common.AbstractWindowsCommand):
    """Extract command history by scanning for _COMMAND_HISTORY"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('MAX_HISTORY', short_option='M', default=MAX_HISTORY_DEFAULT, action='store', type='int', help='CommandCountMax (default = 50)')

    def cmdhistory_process_filter(self, addr_space):
        if False:
            for i in range(10):
                print('nop')
        "Generator for processes that might contain command \n        history information. \n\n        Takes into account if we're on Windows 7 or an earlier\n        operator system. \n\n        @param addr_space: a kernel address space. \n        "
        use_conhost = (6, 1) <= (addr_space.profile.metadata.get('major', 0), addr_space.profile.metadata.get('minor', 0))
        for task in tasks.pslist(addr_space):
            process_name = str(task.ImageFileName).lower()
            if use_conhost and process_name == 'conhost.exe' or (not use_conhost and process_name == 'csrss.exe'):
                yield task

    def calculate(self):
        if False:
            while True:
                i = 10
        'The default pattern we search for, as described by Stevens and Casey, \n        is "2\x00". That\'s because CommandCountMax is a little-endian \n        unsigned short whose default value is 50. However, that value can be \n        changed by right clicking cmd.exe and going to Properties->Options->Cmd History \n        or by calling the API function kernel32!SetConsoleHistoryInfo. Thus \n        you can tweak the search criteria by using the --MAX_HISTORY. \n        '
        addr_space = utils.load_as(self._config)
        MAX_HISTORY = self._config.MAX_HISTORY
        srch_pattern = chr(MAX_HISTORY) + '\x00'
        for task in self.cmdhistory_process_filter(addr_space):
            process_space = task.get_process_address_space()
            for found in task.search_process_memory([srch_pattern], vad_filter=lambda x: x.Length < 1073741824):
                hist = obj.Object('_COMMAND_HISTORY', vm=process_space, offset=found - addr_space.profile.get_obj_offset('_COMMAND_HISTORY', 'CommandCountMax'))
                if hist.is_valid(max_history=MAX_HISTORY):
                    yield (task, hist)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Process', str), ('PID', int), ('History Offset', Address), ('Application', str), ('Flags', str), ('Command Count', int), ('Last Added', str), ('Last Displayed', str), ('First Command', str), ('Command Count Max', int), ('Handle', int), ('Command Number', int), ('Command Offset', Address), ('Command', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for (task, hist) in data:
            pointers = obj.Object('Array', targetType='address', count=hist.CommandCountMax, offset=hist.obj_offset + hist.obj_vm.profile.get_obj_offset('_COMMAND_HISTORY', 'CommandBucket'), vm=hist.obj_vm)
            values = [str(task.ImageFileName), int(task.UniqueProcessId), Address(hist.obj_offset), str(hist.Application.dereference()), str(hist.Flags), int(hist.CommandCount), str(hist.LastAdded), str(hist.LastDisplayed), str(hist.FirstCommand), int(hist.CommandCountMax), int(hist.ProcessHandle)]
            for (i, p) in enumerate(pointers):
                cmd = p.dereference_as('_COMMAND')
                if cmd and str(cmd.Cmd):
                    yield (0, values + [int(i), Address(cmd.obj_offset), str(cmd.Cmd)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        for (task, hist) in data:
            outfd.write('*' * 50 + '\n')
            outfd.write('CommandProcess: {0} Pid: {1}\n'.format(task.ImageFileName, task.UniqueProcessId))
            outfd.write('CommandHistory: {0:#x} Application: {1} Flags: {2}\n'.format(hist.obj_offset, hist.Application.dereference(), hist.Flags))
            outfd.write('CommandCount: {0} LastAdded: {1} LastDisplayed: {2}\n'.format(hist.CommandCount, hist.LastAdded, hist.LastDisplayed))
            outfd.write('FirstCommand: {0} CommandCountMax: {1}\n'.format(hist.FirstCommand, hist.CommandCountMax))
            outfd.write('ProcessHandle: {0:#x}\n'.format(hist.ProcessHandle))
            pointers = obj.Object('Array', targetType='address', count=hist.CommandCountMax, offset=hist.obj_offset + hist.obj_vm.profile.get_obj_offset('_COMMAND_HISTORY', 'CommandBucket'), vm=hist.obj_vm)
            for (i, p) in enumerate(pointers):
                cmd = p.dereference_as('_COMMAND')
                if cmd and str(cmd.Cmd):
                    outfd.write('Cmd #{0} @ {1:#x}: {2}\n'.format(i, cmd.obj_offset, str(cmd.Cmd)))

class Consoles(CmdScan):
    """Extract command history by scanning for _CONSOLE_INFORMATION"""

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        CmdScan.__init__(self, config, *args, **kwargs)
        config.add_option('HISTORY_BUFFERS', short_option='B', default=4, action='store', type='int', help='HistoryBufferMax (default = 4)')

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        srch_pattern = chr(self._config.MAX_HISTORY) + '\x00'
        for task in self.cmdhistory_process_filter(addr_space):
            for found in task.search_process_memory([srch_pattern], vad_filter=lambda x: x.Length < 1073741824):
                console = obj.Object('_CONSOLE_INFORMATION', offset=found - addr_space.profile.get_obj_offset('_CONSOLE_INFORMATION', 'CommandHistorySize'), vm=task.get_process_address_space(), parent=task)
                if console.HistoryBufferMax != self._config.HISTORY_BUFFERS or console.HistoryBufferCount > self._config.HISTORY_BUFFERS:
                    continue
                history = obj.Object('_COMMAND_HISTORY', offset=console.HistoryList.Flink.dereference().obj_offset - addr_space.profile.get_obj_offset('_COMMAND_HISTORY', 'ListEntry'), vm=task.get_process_address_space())
                if history.CommandCountMax != self._config.MAX_HISTORY:
                    continue
                yield (task, console)

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Console Process', str), ('Console PID', int), ('Console ID', int), ('Command History Size', int), ('History Buffer Count', int), ('History Buffer Max', int), ('OriginalTitle', str), ('Title', str), ('Attached Process Name', str), ('Attached Process PID', int), ('Attached Process Handle', int), ('Command History ID', int), ('Command History Applications', str), ('Command History Flags', str), ('Command History Count', int), ('Command History Last Added', str), ('Command History Last Displayed', str), ('Command History First Command', str), ('Command History Command Count Max', int), ('Command History Process Handle', int), ('Command History Command Number', int), ('Command History Command Offset', Address), ('Command History Command String', str), ('EXE Alias', str), ('EXE Alias Source', str), ('EXE Alias Target', str), ('Screen ID', str), ('Screen X', int), ('Screen Y', int), ('Screen Dump', str)], self.generator(data))

    def _get_values(self, task, console, process=None, console_proc=None, hist=None, hist_i=None, hist_cmd=None, exe_alias=None, screen=None):
        if False:
            print('Hello World!')
        v = [str(task.ImageFileName), int(task.UniqueProcessId), int(console.obj_offset), int(console.CommandHistorySize), int(console.HistoryBufferCount), int(console.HistoryBufferMax), str(console.OriginalTitle.dereference()), str(console.Title.dereference())]
        if process is not None and console_proc is not None:
            v.extend([str(process.ImageFileName), int(process.UniqueProcessId), int(console_proc.ProcessHandle)])
        else:
            v.extend(['', -1, -1])
        if hist is not None:
            v.extend([int(hist.obj_offset), str(hist.Application.dereference()), str(hist.Flags), int(hist.CommandCount), str(hist.LastAdded), str(hist.LastDisplayed), str(hist.FirstCommand), int(hist.CommandCountMax), int(hist.ProcessHandle)])
            if hist_i is None or hist_cmd is None:
                v.extend([-1, Address(-1), ''])
            else:
                v.extend([int(hist_i), Address(hist_cmd.obj_offset), str(hist_cmd.Cmd)])
        else:
            v.extend([-1, '', '', -1, '', '', '', -1, -1, -1, Address(-1), ''])
        if exe_alias is not None:
            v.extend([str(exe_alias.ExeName.dereference()), str(alias.Source.dereference()), str(alias.Target.dereference())])
        else:
            v.extend(['', '', ''])
        if screen is not None:
            v.extend([str(screen.dereference()), int(screen.ScreenX), int(screen.ScreenY), '\n'.join(screen.get_buffer())])
        else:
            v.extend(['', -1, -1, ''])
        return v

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (task, console) in data:
            has_yielded = False
            for console_proc in console.get_processes():
                process = console_proc.reference_object_by_handle()
                if process:
                    has_yielded = True
                    yield (0, self._get_values(task, console, process=process, console_proc=console_proc))
            for hist in console.get_histories():
                cmds_processed = False
                for (i, cmd) in hist.get_commands():
                    if cmd.Cmd:
                        cmds_processed = True
                        yield (0, self._get_values(task, console, hist=hist, hist_i=i, hist_cmd=cmd))
                    has_yielded = cmds_processed
                if not cmds_processed:
                    has_yielded = True
                    yield (0, self._get_values(task, console, hist=hist))
            for exe_alias in console.get_exe_aliases():
                for alias in exe_alias.get_aliases():
                    has_yielded = True
                    yield (0, self._get_values(task, console, exe_alias=alias))
            for screen in console.get_screens():
                has_yielded = True
                yield (0, self._get_values(task, console, screen=screen))
            if not has_yielded:
                yield (0, self._get_values(task, console))

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for (task, console) in data:
            outfd.write('*' * 50 + '\n')
            outfd.write('ConsoleProcess: {0} Pid: {1}\n'.format(task.ImageFileName, task.UniqueProcessId))
            outfd.write('Console: {0:#x} CommandHistorySize: {1}\n'.format(console.obj_offset, console.CommandHistorySize))
            outfd.write('HistoryBufferCount: {0} HistoryBufferMax: {1}\n'.format(console.HistoryBufferCount, console.HistoryBufferMax))
            outfd.write('OriginalTitle: {0}\n'.format(console.OriginalTitle.dereference()))
            outfd.write('Title: {0}\n'.format(console.Title.dereference()))
            for console_proc in console.get_processes():
                process = console_proc.reference_object_by_handle()
                if process:
                    outfd.write('AttachedProcess: {0} Pid: {1} Handle: {2:#x}\n'.format(process.ImageFileName, process.UniqueProcessId, console_proc.ProcessHandle))
            for hist in console.get_histories():
                outfd.write('----\n')
                outfd.write('CommandHistory: {0:#x} Application: {1} Flags: {2}\n'.format(hist.obj_offset, hist.Application.dereference(), hist.Flags))
                outfd.write('CommandCount: {0} LastAdded: {1} LastDisplayed: {2}\n'.format(hist.CommandCount, hist.LastAdded, hist.LastDisplayed))
                outfd.write('FirstCommand: {0} CommandCountMax: {1}\n'.format(hist.FirstCommand, hist.CommandCountMax))
                outfd.write('ProcessHandle: {0:#x}\n'.format(hist.ProcessHandle))
                for (i, cmd) in hist.get_commands():
                    if cmd.Cmd:
                        outfd.write('Cmd #{0} at {1:#x}: {2}\n'.format(i, cmd.obj_offset, str(cmd.Cmd)))
            for exe_alias in console.get_exe_aliases():
                for alias in exe_alias.get_aliases():
                    outfd.write('----\n')
                    outfd.write('Alias: {0} Source: {1} Target: {2}\n'.format(exe_alias.ExeName.dereference(), alias.Source.dereference(), alias.Target.dereference()))
            for screen in console.get_screens():
                outfd.write('----\n')
                outfd.write('Screen {0:#x} X:{1} Y:{2}\n'.format(screen.dereference(), screen.ScreenX, screen.ScreenY))
                outfd.write('Dump:\n{0}\n'.format('\n'.join(screen.get_buffer())))