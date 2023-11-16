"""
@author     : Bridgey the Geek
@license    : GPL 2 or later
@contact    : bridgeythegeek@gmail.com
"""
import os
import volatility.debug as debug
import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.plugins.gui.messagehooks as messagehooks
import volatility.win32 as win32
from volatility.renderers import TreeGrid
supported_controls = {'edit': 'COMCTL_EDIT', 'listbox': 'COMCTL_LISTBOX'}
editbox_vtypes_xp_x86 = {'COMCTL_EDIT': [238, {'hBuf': [0, ['unsigned long']], 'hWnd': [56, ['unsigned long']], 'parenthWnd': [88, ['unsigned long']], 'nChars': [12, ['unsigned long']], 'selStart': [20, ['unsigned long']], 'selEnd': [24, ['unsigned long']], 'pwdChar': [48, ['unsigned short']], 'undoBuf': [128, ['unsigned long']], 'undoPos': [132, ['long']], 'undoLen': [136, ['long']], 'bEncKey': [236, ['unsigned char']]}], 'COMCTL_LISTBOX': [64, {'hWnd': [0, ['unsigned long']], 'parenthWnd': [4, ['unsigned long']], 'atomHandle': [8, ['unsigned long']], 'firstVisibleRow': [16, ['unsigned long']], 'caretPos': [20, ['long']], 'rowsVisible': [28, ['unsigned long']], 'itemCount': [32, ['unsigned long']], 'stringsStart': [44, ['unsigned long']], 'stringsLength': [52, ['unsigned long']]}]}
editbox_vtypes_xp_x64 = {'COMCTL_EDIT': [322, {'hBuf': [0, ['unsigned long']], 'hWnd': [64, ['unsigned long']], 'parenthWnd': [96, ['unsigned long']], 'nChars': [16, ['unsigned long']], 'selStart': [24, ['unsigned long']], 'selEnd': [32, ['unsigned long']], 'pwdChar': [52, ['unsigned short']], 'undoBuf': [168, ['address']], 'undoPos': [176, ['long']], 'undoLen': [180, ['long']], 'bEncKey': [320, ['unsigned char']]}], 'COMCTL_LISTBOX': [256, {'hWnd': [0, ['unsigned long']], 'parenthWnd': [8, ['unsigned long']], 'firstVisibleRow': [32, ['unsigned long']], 'caretPos': [40, ['unsigned long']], 'rowsVisible': [44, ['unsigned long']], 'itemCount': [48, ['unsigned long']], 'stringsStart': [64, ['address']], 'stringsLength': [76, ['unsigned long']]}]}
editbox_vtypes_vista7810_x86 = {'COMCTL_EDIT': [246, {'hBuf': [0, ['unsigned long']], 'hWnd': [56, ['unsigned long']], 'parenthWnd': [88, ['unsigned long']], 'nChars': [12, ['unsigned long']], 'selStart': [20, ['unsigned long']], 'selEnd': [24, ['unsigned long']], 'pwdChar': [48, ['unsigned short']], 'undoBuf': [136, ['unsigned long']], 'undoPos': [140, ['long']], 'undoLen': [144, ['long']], 'bEncKey': [244, ['unsigned char']]}], 'COMCTL_LISTBOX': [64, {'hWnd': [0, ['unsigned long']], 'parenthWnd': [4, ['unsigned long']], 'atomHandle': [8, ['unsigned long']], 'firstVisibleRow': [16, ['unsigned long']], 'caretPos': [20, ['long']], 'rowsVisible': [28, ['unsigned long']], 'itemCount': [32, ['unsigned long']], 'stringsStart': [44, ['unsigned long']], 'stringsLength': [52, ['unsigned long']]}]}
editbox_vtypes_vista7810_x64 = {'COMCTL_EDIT': [322, {'hBuf': [0, ['unsigned long']], 'hWnd': [64, ['unsigned long']], 'parenthWnd': [96, ['unsigned long']], 'nChars': [16, ['unsigned long']], 'selStart': [24, ['unsigned long']], 'selEnd': [32, ['unsigned long']], 'pwdChar': [52, ['unsigned short']], 'undoBuf': [168, ['address']], 'undoPos': [176, ['long']], 'undoLen': [180, ['long']], 'bEncKey': [320, ['unsigned char']]}], 'COMCTL_LISTBOX': [84, {'hWnd': [0, ['unsigned long']], 'parenthWnd': [8, ['unsigned long']], 'firstVisibleRow': [32, ['unsigned long']], 'caretPos': [40, ['unsigned long']], 'rowsVisible': [44, ['unsigned long']], 'itemCount': [48, ['unsigned long']], 'stringsStart': [64, ['address']], 'stringsLength': [76, ['unsigned long']]}]}

class COMCTL_EDIT(obj.CType):
    """Methods for the Edit structure"""

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'String representation of the Edit'
        _MAX_OUT = 50
        text = self.get_text(no_crlf=True)
        text = '{}...'.format(text[:_MAX_OUT - 3]) if len(text) > _MAX_OUT else text
        undo = self.get_undo(no_crlf=True)
        undo = '{}...'.format(undo[:_MAX_OUT - 3]) if len(undo) > _MAX_OUT else undo
        return '<{0}(Text="{1}", Len={2}, Pwd={3}, Undo="{4}", UndoLen={5})>'.format(self.__class__.__name__, text, self.nChars, self.is_pwd(), undo, self.undoLen)

    def get_text(self, no_crlf=False):
        if False:
            while True:
                i = 10
        'Get the text from the control\n\n        :param no_crlf:\n        :return:\n        '
        if self.nChars < 1:
            return ''
        text_deref = obj.Object('unsigned long', offset=self.hBuf, vm=self.obj_vm)
        raw = self.obj_vm.read(text_deref, self.nChars * 2)
        if not self.pwdChar == 0:
            raw = COMCTL_EDIT.rtl_run_decode_unicode_string(self.bEncKey, raw)
        if no_crlf:
            return raw.decode('utf-16').replace('\r\n', '.')
        else:
            return raw.decode('utf-16')

    def get_undo(self, no_crlf=False):
        if False:
            while True:
                i = 10
        'Get the contents of the undo buffer\n\n        :param no_crlf:\n        :return:\n        '
        if self.undoLen < 1:
            return ''
        if no_crlf:
            return self.obj_vm.read(self.undoBuf, self.undoLen * 2).decode('utf-16').replace('\r\n', '.')
        else:
            return self.obj_vm.read(self.undoBuf, self.undoLen * 2).decode('utf-16')

    def is_pwd(self):
        if False:
            i = 10
            return i + 15
        'Is this a password control?\n\n        :return:\n        '
        return self.pwdChar != 0

    def dump_meta(self, outfd):
        if False:
            i = 10
            return i + 15
        'Dumps the meta data of the control\n        \n        @param  outfd: \n        '
        outfd.write('nChars            : {}\n'.format(self.nChars))
        outfd.write('selStart          : {}\n'.format(self.selStart))
        outfd.write('selEnd            : {}\n'.format(self.selEnd))
        outfd.write('isPwdControl      : {}\n'.format(self.is_pwd()))
        outfd.write('undoPos           : {}\n'.format(self.undoPos))
        outfd.write('undoLen           : {}\n'.format(self.undoLen))
        outfd.write('address-of undoBuf: {:#x}\n'.format(self.undoBuf))
        outfd.write('undoBuf           : {}\n'.format(self.get_undo(no_crlf=True)))

    def dump_data(self, outfd):
        if False:
            print('Hello World!')
        'Dumps the data of the control\n        \n        @param  outfd: \n        '
        outfd.write('{}\n'.format(self.get_text()))

    @staticmethod
    def rtl_run_decode_unicode_string(key, data):
        if False:
            print('Hello World!')
        s = ''.join([chr(ord(data[i - 1]) ^ ord(data[i]) ^ key) for i in range(1, len(data))])
        s = chr(ord(data[0]) ^ (key | 67)) + s
        return s

class COMCTL_LISTBOX(obj.CType):
    """Methods for the Listbox structure"""

    def __str__(self):
        if False:
            while True:
                i = 10
        'String representation of the Listbox'
        _MAX_OUT = 50
        text = self.get_text(joiner='|')
        text = '{}...'.format(text[:_MAX_OUT - 3]) if len(text) > _MAX_OUT else text
        return '<{0}(Text="{1}", Items={2}, Caret={3}>'.format(self.__class__.__name__, text, self.itemCount, self.caretPos)

    def get_text(self, joiner='\n'):
        if False:
            return 10
        'Get the text from the control\n\n        @param joiner:\n        @return:\n        '
        if self.stringsLength < 1:
            return ''
        raw = self.obj_vm.read(self.stringsStart, self.stringsLength)
        return joiner.join(split_null_strings(raw))

    def dump_meta(self, outfd):
        if False:
            i = 10
            return i + 15
        'Dumps the meta data of the control\n\n        @param  outfd:\n        '
        outfd.write('firstVisibleRow   : {}\n'.format(self.firstVisibleRow))
        outfd.write('caretPos          : {}\n'.format(self.caretPos))
        outfd.write('rowsVisible       : {}\n'.format(self.rowsVisible))
        outfd.write('itemCount         : {}\n'.format(self.itemCount))
        outfd.write('stringsStart      : {:#x}\n'.format(self.stringsStart))
        outfd.write('stringsLength     : {}\n'.format(self.stringsLength))

    def dump_data(self, outfd):
        if False:
            i = 10
            return i + 15
        'Dumps the data of the control\n\n        @param  outfd:\n        '
        outfd.write('{}\n'.format(self.get_text()))

def split_null_strings(data):
    if False:
        print('Hello World!')
    'Splits a concatenation of null-terminated utf-16 strings\n    \n    @param  data:\n    '
    strings = []
    start = 0
    for i in xrange(0, len(data), 2):
        if data[i] == '\x00' and data[i + 1] == '\x00':
            strings.append(data[start:i])
            start = i + 2
    return [s.decode('utf-16') for s in strings]

def dump_to_file(ctrl, pid, proc_name, folder):
    if False:
        while True:
            i = 10
    'Dumps the data of the control to a file\n\n    @param  ctrl:\n    @param  pid:\n    @param  proc_name:\n    @param  folder:\n    '
    ctrl_safe_name = str(ctrl.__class__.__name__).split('_')[-1].lower()
    file_name = '{0}_{1}_{2}_{3:#x}.txt'.format(pid, proc_name, ctrl_safe_name, ctrl.v())
    with open(os.path.join(folder, file_name), 'wb') as out_file:
        out_file.write(ctrl.get_text())

class Editbox(common.AbstractWindowsCommand):
    """Displays information about Edit controls. (Listbox experimental.)"""
    editbox_classes = {'COMCTL_EDIT': COMCTL_EDIT, 'COMCTL_LISTBOX': COMCTL_LISTBOX}

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('PID', short_option='p', default=None, help='Operate on these Process IDs (comma-separated)', action='store', type='str')
        config.add_option('DUMP-DIR', short_option='D', default=None, help='Save the found text to files in this folder', action='store', type='str')
        self.fake_32bit = False

    @staticmethod
    def apply_types(addr_space, meta=None):
        if False:
            return 10
        'Add the correct vtypes and classes for the profile\n\n        @param  addr_space:        \n        @param  meta: \n        '
        if not meta:
            meta = addr_space.profile.metadata
        if meta['os'] == 'windows':
            if meta['major'] == 5:
                if meta['memory_model'] == '32bit':
                    addr_space.profile.vtypes.update(editbox_vtypes_xp_x86)
                elif meta['memory_model'] == '64bit':
                    addr_space.profile.vtypes.update(editbox_vtypes_xp_x64)
                else:
                    debug.error('The selected address space is not supported')
                addr_space.profile.compile()
            elif meta['major'] == 6:
                if meta['memory_model'] == '32bit':
                    addr_space.profile.vtypes.update(editbox_vtypes_vista7810_x86)
                elif meta['memory_model'] == '64bit':
                    addr_space.profile.vtypes.update(editbox_vtypes_vista7810_x64)
                else:
                    debug.error('The selected address space is not supported')
                addr_space.profile.compile()
            else:
                debug.error('The selected address space is not supported')
        else:
            debug.error('The selected address space is not supported')

    def calculate(self):
        if False:
            print('Hello World!')
        'Parse the control structures'
        if self._config.DUMP_DIR and (not os.path.isdir(self._config.dump_dir)):
            debug.error('{0} is not a directory'.format(self._config.dump_dir))
        addr_space = utils.load_as(self._config)
        addr_space.profile.object_classes.update(Editbox.editbox_classes)
        self.apply_types(addr_space)
        tasks = win32.tasks.pslist(addr_space)
        if self._config.PID:
            pids = [int(p) for p in self._config.PID.split(',')]
            the_tasks = [t for t in tasks if t.UniqueProcessId in pids]
        else:
            the_tasks = [t for t in tasks]
        if len(the_tasks) < 1:
            return
        mh = messagehooks.MessageHooks(self._config)
        for (winsta, atom_tables) in mh.calculate():
            for desktop in winsta.desktops():
                for (wnd, _level) in desktop.windows(desktop.DeskInfo.spwnd):
                    if wnd.Process in the_tasks:
                        atom_class = mh.translate_atom(winsta, atom_tables, wnd.ClassAtom)
                        if atom_class:
                            atom_class = str(atom_class)
                            if '!' in atom_class:
                                comctl_class = atom_class.split('!')[-1].lower()
                                if comctl_class in supported_controls:
                                    if wnd.Process.IsWow64 and (not self.fake_32bit):
                                        meta = addr_space.profile.metadata
                                        meta['memory_model'] = '32bit'
                                        self.apply_types(addr_space, meta)
                                        self.fake_32bit = True
                                    elif not wnd.Process.IsWow64 and self.fake_32bit:
                                        self.apply_types(addr_space)
                                        self.fake_32bit = False
                                    context = '{0}\\{1}\\{2}'.format(winsta.dwSessionId, winsta.Name, desktop.Name)
                                    task_vm = wnd.Process.get_process_address_space()
                                    wndextra_offset = wnd.v() + addr_space.profile.get_obj_size('tagWND')
                                    wndextra = obj.Object('address', offset=wndextra_offset, vm=task_vm)
                                    ctrl = obj.Object(supported_controls[comctl_class], offset=wndextra, vm=task_vm)
                                    if self._config.DUMP_DIR:
                                        dump_to_file(ctrl, wnd.Process.UniqueProcessId, wnd.Process.ImageFileName, self._config.DUMP_DIR)
                                    yield (context, atom_class, wnd.Process.UniqueProcessId, wnd.Process.ImageFileName, wnd.Process.IsWow64, ctrl)

    def render_table(self, outfd, data):
        if False:
            i = 10
            return i + 15
        'Output the results as a table\n        \n        @param  outfd: <file>\n        @param  data: <generator>\n        '
        self.table_header(outfd, [('PID', '6'), ('Process', '14'), ('Control', '')])
        for (context, atom_class, pid, proc_name, is_wow64, ctrl) in data:
            self.table_row(outfd, pid, proc_name, str(ctrl))

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Wnd Context', str), ('Process ID', int), ('ImageFileName', str), ('IsWow64', str), ('atom_class', str), ('value-of WndExtra', str), ('nChars', int), ('selStart', int), ('selEnd', int), ('isPwdControl', int), ('undoPos', int), ('undoLen', int), ('address-of undoBuf', str), ('undoBuf', str), ('Data', str)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (context, atom_class, pid, proc_name, is_wow64, ctrl) in data:
            yield (0, [str(context), int(pid), str(proc_name), str('Yes' if is_wow64 else 'No'), str(atom_class), str(hex(int(ctrl.v()))), int(ctrl.nChars), int(ctrl.selStart), int(ctrl.is_pwd()), int(ctrl.undoPos), int(ctrl.undoLen), int(ctrl.selEnd), str(ctrl.undoBuf), str(ctrl.get_undo(no_crlf=True)), str(ctrl.get_text())])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        'Output the results as a text report\n        \n        @param  outfd: <file>\n        @param  data: <generator>\n        '
        for (context, atom_class, pid, proc_name, is_wow64, ctrl) in data:
            outfd.write('{}\n'.format('*' * 30))
            outfd.write('Wnd Context       : {}\n'.format(context))
            outfd.write('Process ID        : {}\n'.format(pid))
            outfd.write('ImageFileName     : {}\n'.format(proc_name))
            outfd.write('IsWow64           : {}\n'.format('Yes' if is_wow64 else 'No'))
            outfd.write('atom_class        : {}\n'.format(atom_class))
            outfd.write('value-of WndExtra : {:#x}\n'.format(ctrl.v()))
            ctrl.dump_meta(outfd)
            outfd.write('{}\n'.format('-' * 25))
            ctrl.dump_data(outfd)