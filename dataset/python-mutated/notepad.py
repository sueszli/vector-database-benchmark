import os
import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.taskmods as taskmods
from volatility.renderers import TreeGrid

class _HEAP(obj.CType):
    """ A Heap on XP and 2003 """

    def is_valid(self):
        if False:
            while True:
                i = 10
        return obj.CType.is_valid(self) and self.Signature == 4009750271

    def segments(self):
        if False:
            return 10
        ' A list of the _HEAP_SEGMENTs. \n\n        This is an array of pointers so we have to deref\n        before returning or the caller will be calling \n        is_valid on the pointer and not the object. \n        '
        return [seg.dereference() for seg in self.Segments if seg != 0]

class _HEAP_SEGMENT(obj.CType):
    """ A Heap Segment on XP and 2003 """

    def is_valid(self):
        if False:
            i = 10
            return i + 15
        return obj.CType.is_valid(self) and self.Signature == 4293853166

    def heap_entries(self):
        if False:
            return 10
        'Enumerate the heaps in this segment. \n\n        ##FIXME: \n        * Raise ValueError if corruptions are detected. \n        * Should we start at FirstEntry or Entry?\n        '
        next = self.Entry
        last = self.LastValidEntry.dereference()
        chunk_size = self.obj_vm.profile.get_obj_size('_HEAP_ENTRY')
        while next and next.obj_offset < last.obj_offset:
            yield next
            next = obj.Object('_HEAP_ENTRY', offset=next.obj_offset + next.Size * chunk_size, vm=next.obj_vm)

class _HEAP_ENTRY(obj.CType):
    """ A Heap Entry """

    def get_data(self):
        if False:
            print('Hello World!')
        chunk_size = self.obj_vm.profile.get_obj_size('_HEAP_ENTRY')
        return self.obj_vm.zread(self.obj_offset + chunk_size, self.Size * chunk_size)

    def get_extra(self):
        if False:
            while True:
                i = 10
        chunk_size = self.obj_vm.profile.get_obj_size('_HEAP_ENTRY')
        return obj.Object('_HEAP_ENTRY_EXTRA', offset=self.obj_offset + chunk_size * (self.Size - 1), vm=self.obj_vm)

class XPHeapModification(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5, 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            return 10
        heap_flags = {'HEAP_NO_SERIALIZE': 0, 'HEAP_GROWABLE': 1, 'HEAP_GENERATE_EXCEPTIONS': 2, 'HEAP_ZERO_MEMORY': 3, 'HEAP_REALLOC_IN_PLACE_ONLY': 4, 'HEAP_TAIL_CHECKING_ENABLED': 5, 'HEAP_FREE_CHECKING_ENABLED': 6, 'HEAP_DISABLE_COALESCE_ON_FREE': 7, 'HEAP_SETTABLE_USER_VALUE': 8, 'HEAP_CREATE_ALIGN_16': 16, 'HEAP_CREATE_ENABLE_TRACING': 17, 'HEAP_CREATE_ENABLE_EXECUTE': 18, 'HEAP_FLAG_PAGE_ALLOCS': 24, 'HEAP_PROTECTION_ENABLED': 25, 'HEAP_CAPTURE_STACK_BACKTRACES': 27, 'HEAP_SKIP_VALIDATION_CHECKS': 28, 'HEAP_VALIDATE_ALL_ENABLED': 29, 'HEAP_VALIDATE_PARAMETERS_ENABLED': 30, 'HEAP_LOCK_USER_ALLOCATED': 31}
        entry_flags = {'busy': 0, 'extra': 1, 'fill': 2, 'virtual': 3, 'last': 4, 'flag1': 5, 'flag2': 6, 'flag3': 7}
        profile.merge_overlay({'_HEAP': [None, {'Flags': [None, ['Flags', {'bitmap': heap_flags}]], 'ForceFlags': [None, ['Flags', {'bitmap': heap_flags}]]}], '_HEAP_FREE_ENTRY': [None, {'Flags': [None, ['Flags', {'target': 'unsigned char', 'bitmap': entry_flags}]]}], '_HEAP_ENTRY': [None, {'Flags': [None, ['Flags', {'target': 'unsigned char', 'bitmap': entry_flags}]]}], '_HEAP_SEGMENT': [None, {'Flags': [None, ['Flags', {'bitmap': {'HEAP_USER_ALLOCATED': 0}}]]}]})
        profile.object_classes.update({'_HEAP_ENTRY': _HEAP_ENTRY, '_HEAP': _HEAP, '_HEAP_SEGMENT': _HEAP_SEGMENT})

class Notepad(taskmods.DllList):
    """List currently displayed notepad text"""

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        config.add_option('DUMP-DIR', short_option='D', default=None, help='Dump binary data to this directory')

    @staticmethod
    def is_valid_profile(profile):
        if False:
            return 10
        return profile.metadata.get('os', 'unknown') == 'windows' and profile.metadata.get('major', 0) == 5

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Process', str), ('PID', int), ('Text', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            if str(task.ImageFileName).lower() != 'notepad.exe':
                continue
            process_id = task.UniqueProcessId
            entry_size = task.obj_vm.profile.get_obj_size('_HEAP_ENTRY')
            heap = task.Peb.ProcessHeap.dereference_as('_HEAP')
            for segment in heap.segments():
                for entry in segment.heap_entries():
                    if 'extra' not in str(entry.Flags):
                        continue
                    text = obj.Object('String', offset=entry.obj_offset + entry_size, vm=task.get_process_address_space(), length=entry.Size * entry_size, encoding='utf16')
                    if not text or len(text) == 0:
                        continue
                    else:
                        display_text = text
            yield (0, ['notepad.exe', int(process_id), str(display_text)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for task in data:
            if str(task.ImageFileName).lower() != 'notepad.exe':
                continue
            outfd.write('Process: {0}\n'.format(task.UniqueProcessId))
            entry_size = task.obj_vm.profile.get_obj_size('_HEAP_ENTRY')
            heap = task.Peb.ProcessHeap.dereference_as('_HEAP')
            for segment in heap.segments():
                for entry in segment.heap_entries():
                    if 'extra' not in str(entry.Flags):
                        continue
                    text = obj.Object('String', offset=entry.obj_offset + entry_size, vm=task.get_process_address_space(), length=entry.Size * entry_size, encoding='utf16')
                    if not text or len(text) == 0:
                        continue
                    if self._config.DUMP_DIR:
                        name = 'notepad.{0}.txt'.format(task.UniqueProcessId)
                        path = os.path.join(self._config.DUMP_DIR, name)
                        with open(path, 'wb') as handle:
                            handle.write(entry.get_data())
                        outfd.write('Dumped To: {0}\n'.format(path))
                    outfd.write('Text:\n{0}\n\n'.format(text))