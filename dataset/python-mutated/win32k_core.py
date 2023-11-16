import volatility.obj as obj
import volatility.plugins.gui.constants as consts
import volatility.plugins.overlays.windows.windows as windows
import volatility.utils as utils
import volatility.addrspace as addrspace
import volatility.conf as conf
import volatility.win32.modules as modules

class _MM_SESSION_SPACE(obj.CType):
    """A class for session spaces"""

    def processes(self):
        if False:
            i = 10
            return i + 15
        'Generator for processes in this session. \n    \n        A process is always associated with exactly\n        one session.\n        '
        for p in self.ProcessList.list_of_type('_EPROCESS', 'SessionProcessLinks'):
            if not p.is_valid():
                break
            yield p

    @property
    def Win32KBase(self):
        if False:
            i = 10
            return i + 15
        "Get the base address of the win32k.sys as mapped \n        into this session's memory. \n\n        Since win32k.sys is always the first image to be \n        mapped, we can just grab the first list entry.\n\n        Update: we no longer use the session image list, because\n        it seems to have gone away in Win8/2012."
        for mod in modules.lsmod(self.obj_vm):
            if str(mod.BaseDllName or '').lower() == 'win32k.sys':
                return mod.DllBase
        return obj.Object('Cannot find win32k.sys base address')

    def images(self):
        if False:
            return 10
        "Generator for images (modules) loaded into \n        this session's space"
        metadata = self.obj_vm.profile.metadata
        version = (metadata.get('major', 0), metadata.get('minor', 0))
        if version >= (6, 2):
            raise StopIteration
        else:
            for i in self.ImageList.list_of_type('_IMAGE_ENTRY_IN_SESSION', 'Link'):
                yield i

    def _section_chunks(self, sec_name):
        if False:
            while True:
                i = 10
        'Get the win32k.sys section as an array of \n        32-bit unsigned longs. \n\n        @param sec_name: name of the PE section in win32k.sys \n        to search for. \n\n        @returns all chunks on a 4-byte boundary. \n        '
        dos_header = obj.Object('_IMAGE_DOS_HEADER', offset=self.Win32KBase, vm=self.obj_vm)
        if dos_header:
            try:
                nt_header = dos_header.get_nt_header()
                sections = [sec for sec in nt_header.get_sections() if str(sec.Name) == sec_name]
                if sections:
                    desired_section = sections[0]
                    return obj.Object('Array', targetType='unsigned long', offset=desired_section.VirtualAddress + dos_header.obj_offset, count=desired_section.Misc.VirtualSize / 4, vm=self.obj_vm)
            except ValueError:
                pass
        if not self.Win32KBase:
            return []
        data = self.obj_vm.zread(self.Win32KBase, 5242880)
        buffer_as = addrspace.BufferAddressSpace(conf.ConfObject(), data=data, base_offset=self.Win32KBase)
        return obj.Object('Array', targetType='unsigned long', offset=self.Win32KBase, count=len(data) / 4, vm=buffer_as)

    def find_gahti(self):
        if False:
            return 10
        "Find this session's gahti. \n\n        This can potentially be much faster by searching for \n        '\x00' * sizeof(tagHANDLETYPEINFO) instead \n        of moving on a dword aligned boundary through\n        the section. \n        "
        for chunk in self._section_chunks('.rdata'):
            if not chunk.is_valid():
                continue
            gahti = obj.Object('gahti', offset=chunk.obj_offset, vm=self.obj_vm)
            if str(gahti.types[0].dwAllocTag) == '' and gahti.types[0].bObjectCreateFlags == 0 and (str(gahti.types[1].dwAllocTag) == 'Uswd'):
                return gahti
        return obj.NoneObject('Cannot find win32k!_gahti')

    def find_shared_info(self):
        if False:
            for i in range(10):
                print('nop')
        "Find this session's tagSHAREDINFO structure. \n\n        This structure is embedded in win32k's .data section, \n        (i.e. not in dynamically allocated memory). Thus we \n        iterate over each DWORD-aligned possibility and treat \n        it as a tagSHAREDINFO until the sanity checks are met. \n        "
        for chunk in self._section_chunks('.data'):
            if not chunk.is_valid():
                continue
            shared_info = obj.Object('tagSHAREDINFO', offset=chunk.obj_offset, vm=self.obj_vm)
            try:
                if shared_info.is_valid():
                    return shared_info
            except obj.InvalidOffsetError:
                pass
        return obj.NoneObject('Cannot find win32k!gSharedInfo')

class tagSHAREDINFO(obj.CType):
    """A class for shared info blocks"""

    def is_valid(self):
        if False:
            return 10
        'The sanity checks for tagSHAREDINFO structures'
        if not obj.CType.is_valid(self):
            return False
        if self.ulSharedDelta != 0:
            return False
        if not self.psi.is_valid():
            return False
        if self.psi.cbHandleTable == None:
            return False
        if self.psi.cbHandleTable < 4096:
            return False
        return self.psi.cbHandleTable / self.obj_vm.profile.get_obj_size('_HANDLEENTRY') == self.psi.cHandleEntries

    def handles(self, filters=None):
        if False:
            print('Hello World!')
        'Carve handles from the shared info block. \n\n        @param filters: a list of callables that perform\n        checks and return True if the handle should be\n        included in output.\n        '
        if filters == None:
            filters = []
        hnds = obj.Object('Array', targetType='_HANDLEENTRY', offset=self.aheList, vm=self.obj_vm, count=self.psi.cHandleEntries)
        for (i, h) in enumerate(hnds):
            if not h.Free:
                if h.phead.h != h.wUniq << 16 | 65535 & i:
                    continue
            b = False
            for filt in filters:
                if not filt(h):
                    b = True
                    break
            if not b:
                yield h

class _HANDLEENTRY(obj.CType):
    """A for USER handle entries"""

    def reference_object(self):
        if False:
            while True:
                i = 10
        "Reference the object this handle represents. \n\n        If the object's type is not in our map, we don't know\n        what type of object to instantiate so its filled with\n        obj.NoneObject() instead. \n        "
        object_map = dict(TYPE_WINDOW='tagWND', TYPE_HOOK='tagHOOK', TYPE_CLIPDATA='tagCLIPDATA', TYPE_WINEVENTHOOK='tagEVENTHOOK', TYPE_TIMER='tagTIMER')
        object_type = object_map.get(str(self.bType), None)
        if not object_type:
            return obj.NoneObject('Cannot reference object type')
        return obj.Object(object_type, offset=self.phead, vm=self.obj_vm)

    @property
    def Free(self):
        if False:
            return 10
        'Check if the handle has been freed'
        return str(self.bType) == 'TYPE_FREE'

    @property
    def ThreadOwned(self):
        if False:
            return 10
        'Handles of these types are always thread owned'
        return str(self.bType) in ['TYPE_WINDOW', 'TYPE_SETWINDOWPOS', 'TYPE_HOOK', 'TYPE_DDEACCESS', 'TYPE_DDECONV', 'TYPE_DDEXACT', 'TYPE_WINEVENTHOOK', 'TYPE_INPUTCONTEXT', 'TYPE_HIDDATA', 'TYPE_TOUCH', 'TYPE_GESTURE']

    @property
    def ProcessOwned(self):
        if False:
            print('Hello World!')
        'Handles of these types are always process owned'
        return str(self.bType) in ['TYPE_MENU', 'TYPE_CURSOR', 'TYPE_TIMER', 'TYPE_CALLPROC', 'TYPE_ACCELTABLE']

    @property
    def Thread(self):
        if False:
            print('Hello World!')
        'Return the ETHREAD if its thread owned'
        if self.ThreadOwned:
            return self.pOwner.dereference_as('tagTHREADINFO').pEThread.dereference()
        return obj.NoneObject('Cannot find thread')

    @property
    def Process(self):
        if False:
            i = 10
            return i + 15
        'Return the _EPROCESS if its process or thread owned'
        if self.ProcessOwned:
            return self.pOwner.dereference_as('tagPROCESSINFO').Process.dereference()
        elif self.ThreadOwned:
            return self.pOwner.dereference_as('tagTHREADINFO').ppi.Process.dereference()
        return obj.NoneObject('Cannot find process')

class tagWINDOWSTATION(obj.CType, windows.ExecutiveObjectMixin):
    """A class for Windowstation objects"""

    def is_valid(self):
        if False:
            return 10
        return obj.CType.is_valid(self) and self.dwSessionId < 255

    @property
    def PhysicalAddress(self):
        if False:
            i = 10
            return i + 15
        "This is a simple wrapper to always return the object's\n        physical offset regardless of what AS its instantiated in"
        if hasattr(self.obj_vm, 'vtop'):
            return self.obj_vm.vtop(self.obj_offset)
        else:
            return self.obj_offset

    @property
    def LastRegisteredViewer(self):
        if False:
            print('Hello World!')
        'The EPROCESS of the last registered \n        clipboard viewer'
        return self.spwndClipViewer.head.pti.ppi.Process

    @property
    def AtomTable(self):
        if False:
            return 10
        'This atom table belonging to this window \n        station object'
        return self.pGlobalAtomTable.dereference_as('_RTL_ATOM_TABLE')

    @property
    def Interactive(self):
        if False:
            print('Hello World!')
        'Check if a window station is interactive'
        return not self.dwWSF_Flags & 4

    @property
    def Name(self):
        if False:
            i = 10
            return i + 15
        'Get the window station name. \n\n        Since window stations are securable objects, \n        and are managed by the same object manager as\n        processes, threads, etc, there is an object\n        header which stores the name.\n        '
        object_hdr = obj.Object('_OBJECT_HEADER', vm=self.obj_vm, offset=self.obj_offset - self.obj_vm.profile.get_obj_offset('_OBJECT_HEADER', 'Body'), native_vm=self.obj_native_vm)
        return str(object_hdr.NameInfo.Name or '')

    def traverse(self):
        if False:
            for i in range(10):
                print('nop')
        'A generator that yields window station objects'
        yield self
        nextwinsta = self.rpwinstaNext.dereference()
        while nextwinsta.is_valid() and nextwinsta.v() != 0:
            yield nextwinsta
            nextwinsta = nextwinsta.rpwinstaNext.dereference()

    def desktops(self):
        if False:
            for i in range(10):
                print('nop')
        "A generator that yields the window station's desktops"
        desk = self.rpdeskList.dereference()
        while desk.is_valid() and desk.v() != 0 and desk.Name:
            yield desk
            desk = desk.rpdeskNext.dereference()

class tagDESKTOP(tagWINDOWSTATION):
    """A class for Desktop objects"""

    def is_valid(self):
        if False:
            for i in range(10):
                print('nop')
        return obj.CType.is_valid(self) and self.dwSessionId < 255

    @property
    def WindowStation(self):
        if False:
            print('Hello World!')
        "Returns this desktop's parent window station"
        return self.rpwinstaParent.dereference()

    @property
    def DeskInfo(self):
        if False:
            return 10
        'Returns the desktop info object'
        return self.pDeskInfo.dereference()

    def threads(self):
        if False:
            print('Hello World!')
        'Generator for _EPROCESS objects attached to this desktop'
        for ti in self.PtiList.list_of_type('tagTHREADINFO', 'PtiLink'):
            if ti.ppi.Process.is_valid():
                yield ti

    def hook_params(self):
        if False:
            return 10
        ' Parameters for the hooks() method.\n\n        These are split out into a function so it can be \n        subclassed by tagTHREADINFO.\n        '
        return (self.DeskInfo.fsHooks, self.DeskInfo.aphkStart)

    def hooks(self):
        if False:
            while True:
                i = 10
        'Generator for tagHOOK info. \n        \n        Hooks are carved using the same algorithm, but different\n        starting points for desktop hooks and thread hooks. Thus\n        the algorithm is presented in this function and the starting\n        point is acquired by calling hook_params (which is then sub-\n        classed by tagTHREADINFO. \n        '
        (fshooks, aphkstart) = self.hook_params()
        WHF_FROM_WH = lambda x: 1 << x + 1
        for (pos, (name, value)) in enumerate(consts.MESSAGE_TYPES):
            if fshooks & WHF_FROM_WH(value):
                hook = aphkstart[pos].dereference()
                for hook in hook.traverse():
                    yield (name, hook)

    def windows(self, win, filter=lambda x: True, level=0):
        if False:
            print('Hello World!')
        'Traverses windows in their Z order, bottom to top.\n\n        @param win: an HWND to start. Usually this is the desktop \n        window currently in focus. \n\n        @param filter: a callable (usually lambda) to use for filtering\n        the results. See below for examples:\n\n        # only print subclassed windows\n        filter = lambda x : x.lpfnWndProc == x.pcls.lpfnWndProc\n\n        # only print processes named csrss.exe\n        filter = lambda x : str(x.head.pti.ppi.Process.ImageFileName).lower()                                 == "csrss.exe" if x.head.pti.ppi else False\n\n        # only print processes by pid\n        filter = lambda x : x.head.pti.pEThread.Cid.UniqueThread == 0x1020\n\n        # only print visible windows\n        filter = lambda x : \'WS_VISIBLE\' not in x.get_flags() \n        '
        seen = set()
        wins = []
        cur = win
        while cur.is_valid() and cur.v() != 0:
            if cur.obj_offset in seen:
                break
            seen.add(cur.obj_offset)
            wins.append(cur)
            cur = cur.spwndNext.dereference()
        while wins:
            cur = wins.pop()
            if not filter(cur):
                continue
            yield (cur, level)
            if cur.spwndChild.is_valid() and cur.spwndChild.v() != 0:
                for (xwin, xlevel) in self.windows(cur.spwndChild, filter=filter, level=level + 1):
                    if xwin.obj_offset in seen:
                        break
                    yield (xwin, xlevel)
                    seen.add(xwin.obj_offset)

    def heaps(self):
        if False:
            i = 10
            return i + 15
        'Generator for the desktop heaps'
        for segment in self.pheapDesktop.Heap.segments():
            for entry in segment.heap_entries():
                yield entry

    def traverse(self):
        if False:
            i = 10
            return i + 15
        'Generator for next desktops in the list'
        yield self
        nextdesk = self.rpdeskNext.dereference()
        while nextdesk.is_valid() and nextdesk.v() != 0:
            yield nextdesk
            nextdesk = nextdesk.rpdeskNext.dereference()

class tagWND(obj.CType):
    """A class for window structures"""

    @property
    def IsClipListener(self):
        if False:
            i = 10
            return i + 15
        'Check if this window listens to clipboard changes'
        return self.bClipboardListener.v()

    @property
    def ClassAtom(self):
        if False:
            for i in range(10):
                print('nop')
        'The class atom for this window'
        return self.pcls.atomClassName

    @property
    def SuperClassAtom(self):
        if False:
            while True:
                i = 10
        "The window's super class"
        return self.pcls.atomNVClassName

    @property
    def Process(self):
        if False:
            print('Hello World!')
        'The EPROCESS that owns the window'
        return self.head.pti.ppi.Process.dereference()

    @property
    def Thread(self):
        if False:
            i = 10
            return i + 15
        'The ETHREAD that owns the window'
        return self.head.pti.pEThread.dereference()

    @property
    def Visible(self):
        if False:
            i = 10
            return i + 15
        'Is this window visible on the desktop'
        return 'WS_VISIBLE' in self.style

    def _get_flags(self, member, flags):
        if False:
            for i in range(10):
                print('nop')
        if flags.has_key(member):
            return flags[member]
        return ','.join([n for (n, v) in flags.items() if member & v == v])

    @property
    def style(self):
        if False:
            for i in range(10):
                print('nop')
        'The basic style flags as a string'
        return self._get_flags(self.m('style').v(), consts.WINDOW_STYLES)

    @property
    def ExStyle(self):
        if False:
            for i in range(10):
                print('nop')
        'The extended style flags as a string'
        return self._get_flags(self.m('ExStyle').v(), consts.WINDOW_STYLES_EX)

class tagRECT(obj.CType):
    """A class for window rects"""

    def get_tup(self):
        if False:
            while True:
                i = 10
        "Return a tuple of the rect's coordinates"
        return (self.left, self.top, self.right, self.bottom)

class tagCLIPDATA(obj.CType):
    """A class for clipboard objects"""

    def as_string(self, fmt):
        if False:
            print('Hello World!')
        'Format the clipboard data as a string. \n\n        @param fmt: the clipboard format. \n\n        Note: we cannot simply override __str__ for this\n        purpose, because the clipboard format is not a member \n        of (or in a parent-child relationship with) the \n        tagCLIPDATA structure, so we must pass it in as \n        an argument. \n        '
        if fmt == 'CF_UNICODETEXT':
            encoding = 'utf16'
        else:
            encoding = 'utf8'
        return obj.Object('String', offset=self.abData.obj_offset, vm=self.obj_vm, encoding=encoding, length=self.cbData)

    def as_hex(self):
        if False:
            for i in range(10):
                print('nop')
        'Format the clipboard contents as a hexdump'
        data = ''.join([chr(c) for c in self.abData])
        return ''.join(['{0:#x}  {1:<48}  {2}\n'.format(self.abData.obj_offset + o, h, ''.join(c)) for (o, h, c) in utils.Hexdump(data)])

class tagTHREADINFO(tagDESKTOP):
    """A class for thread information objects"""

    def get_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Parameters for the _hooks() function'
        return (self.fsHooks, self.aphkStart)

class tagHOOK(obj.CType):
    """A class for message hooks"""

    def traverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Find the next hook in a chain'
        hook = self
        while hook.is_valid() and hook.v() != 0:
            yield hook
            hook = hook.phkNext.dereference()

class tagEVENTHOOK(obj.CType):
    """A class for event hooks"""

    @property
    def dwFlags(self):
        if False:
            for i in range(10):
                print('nop')
        "Event hook flags need special handling so we can't use vtypes"
        f = self.m('dwFlags') >> 1
        flags = [name for (val, name) in consts.EVENT_FLAGS.items() if f & val == val]
        return '|'.join(flags)

class _RTL_ATOM_TABLE(tagWINDOWSTATION):
    """A class for atom tables"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        'Give ourselves an atom cache for quick lookups'
        self.atom_cache = {}
        tagWINDOWSTATION.__init__(self, *args, **kwargs)

    def is_valid(self):
        if False:
            print('Hello World!')
        'Check for validity based on the atom table signature\n        and the maximum allowed number of buckets'
        return obj.CType.is_valid(self) and self.Signature == 1836020801 and (self.NumBuckets < 65535)

    @property
    def NumBuckets(self):
        if False:
            i = 10
            return i + 15
        'Dynamically retrieve the number of atoms in the hash table. \n        First we take into account the offset from the current profile\n        but if it fails and the profile is Win7SP1x64 then we auto set \n        it to the value found in the recently patched versions.\n\n        This is a temporary fix until we have support better support\n        for parsing pdb symbols on the fly. '
        if self.m('NumBuckets') < 65535:
            return self.m('NumBuckets')
        profile = self.obj_vm.profile
        meta = profile.metadata
        major = meta.get('major', 0)
        minor = meta.get('minor', 0)
        build = meta.get('build', 0)
        vers = (major, minor, build)
        if meta.get('memory_model') != '64bit' or vers != (6, 1, 7601):
            return self.m('NumBuckets')
        offset = profile.get_obj_offset('_RTL_ATOM_TABLE', 'NumBuckets')
        number = obj.Object('unsigned long', offset=self.obj_offset + offset + 64, vm=self.obj_vm)
        return number

    def atoms(self):
        if False:
            print('Hello World!')
        'Carve all atoms out of this atom table'
        for bkt in self.Buckets:
            seen = []
            cur = bkt.dereference()
            while cur.is_valid() and cur.v() != 0:
                if cur.obj_offset in seen:
                    break
                yield cur
                seen.append(cur.obj_offset)
                cur = cur.HashLink.dereference()

    def find_atom(self, atom_to_find):
        if False:
            return 10
        'Find an atom by its ID. \n\n        @param atom_to_find: the atom ID (ushort) to find\n\n        @returns an _RTL_ATOM_TALE_ENTRY object \n        '
        if self.atom_cache:
            return self.atom_cache.get(atom_to_find.v(), None)
        self.atom_cache = dict(((atom.Atom.v(), atom) for atom in self.atoms()))
        return self.atom_cache.get(atom_to_find.v(), None)

class _RTL_ATOM_TABLE_ENTRY(obj.CType):
    """A class for atom table entries"""

    @property
    def Pinned(self):
        if False:
            i = 10
            return i + 15
        'Returns True if the atom is pinned'
        return self.Flags == 1

    def is_string_atom(self):
        if False:
            while True:
                i = 10
        'Returns True if the atom is a string atom \n        based on its atom ID. \n        \n        A string atom has ID 0xC000 - 0xFFFF\n        '
        return self.Atom >= 49152 and self.Atom <= 65535

    def is_valid(self):
        if False:
            while True:
                i = 10
        'Perform some sanity checks on the Atom'
        if not obj.CType.is_valid(self):
            return False
        if self.Flags not in (0, 1):
            return False
        return self.NameLength <= 255

class Win32KCoreClasses(obj.ProfileModification):
    """Apply the core object classes"""
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.object_classes.update({'tagWINDOWSTATION': tagWINDOWSTATION, 'tagDESKTOP': tagDESKTOP, '_RTL_ATOM_TABLE': _RTL_ATOM_TABLE, '_RTL_ATOM_TABLE_ENTRY': _RTL_ATOM_TABLE_ENTRY, 'tagTHREADINFO': tagTHREADINFO, 'tagHOOK': tagHOOK, '_LARGE_UNICODE_STRING': windows._UNICODE_STRING, 'tagWND': tagWND, '_MM_SESSION_SPACE': _MM_SESSION_SPACE, 'tagSHAREDINFO': tagSHAREDINFO, '_HANDLEENTRY': _HANDLEENTRY, 'tagEVENTHOOK': tagEVENTHOOK, 'tagRECT': tagRECT, 'tagCLIPDATA': tagCLIPDATA})

class Win32KGahtiVType(obj.ProfileModification):
    """Apply a vtype for win32k!gahti. Adjust the number of 
    handles according to the OS version"""
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        version = (profile.metadata.get('major', 0), profile.metadata.get('minor', 0))
        if version >= (6, 1):
            num_handles = len(consts.HANDLE_TYPE_ENUM_SEVEN)
        else:
            num_handles = len(consts.HANDLE_TYPE_ENUM)
        profile.vtypes.update({'gahti': [None, {'types': [0, ['array', num_handles, ['tagHANDLETYPEINFO']]]}]})

class AtomTablex86Overlay(obj.ProfileModification):
    """Apply the atom table overlays for all x86 Windows"""
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'_RTL_ATOM_TABLE': [None, {'Signature': [0, ['unsigned long']], 'NumBuckets': [12, ['unsigned long']], 'Buckets': [16, ['array', lambda x: x.NumBuckets, ['pointer', ['_RTL_ATOM_TABLE_ENTRY']]]]}], '_RTL_ATOM_TABLE_ENTRY': [None, {'Name': [None, ['String', dict(encoding='utf16', length=lambda x: x.NameLength * 2)]]}]})

class AtomTablex64Overlay(obj.ProfileModification):
    """Apply the atom table overlays for all x64 Windows"""
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.merge_overlay({'_RTL_ATOM_TABLE': [None, {'Signature': [0, ['unsigned long']], 'NumBuckets': [24, ['unsigned long']], 'Buckets': [32, ['array', lambda x: x.NumBuckets, ['pointer', ['_RTL_ATOM_TABLE_ENTRY']]]]}], '_RTL_ATOM_TABLE_ENTRY': [None, {'Name': [None, ['String', dict(encoding='utf16', length=lambda x: x.NameLength * 2)]]}]})

class XP2003x86TimerVType(obj.ProfileModification):
    """Apply the tagTIMER for XP and 2003 x86"""
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x < 6}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.vtypes.update({'tagTIMER': [None, {'head': [0, ['_HEAD']], 'ListEntry': [8, ['_LIST_ENTRY']], 'pti': [16, ['pointer', ['tagTHREADINFO']]], 'spwnd': [20, ['pointer', ['tagWND']]], 'nID': [24, ['unsigned short']], 'cmsCountdown': [28, ['unsigned int']], 'cmsRate': [32, ['unsigned int']], 'flags': [36, ['Flags', {'bitmap': consts.TIMER_FLAGS}]], 'pfn': [40, ['pointer', ['void']]]}]})

class XP2003x64TimerVType(obj.ProfileModification):
    """Apply the tagTIMER for XP and 2003 x64"""
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x < 6}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update({'tagTIMER': [None, {'head': [0, ['_HEAD']], 'ListEntry': [24, ['_LIST_ENTRY']], 'spwnd': [40, ['pointer', ['tagWND']]], 'pti': [32, ['pointer', ['tagTHREADINFO']]], 'nID': [48, ['unsigned short']], 'cmsCountdown': [56, ['unsigned int']], 'cmsRate': [60, ['unsigned int']], 'flags': [64, ['Flags', {'bitmap': consts.TIMER_FLAGS}]], 'pfn': [72, ['pointer', ['void']]]}]})

class Win32Kx86VTypes(obj.ProfileModification):
    """Applies to all x86 windows profiles. 

    These are vtypes not included in win32k.sys PDB.
    """
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit'}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update({'tagWIN32HEAP': [None, {'Heap': [0, ['_HEAP']]}], 'tagCLIPDATA': [None, {'cbData': [8, ['unsigned int']], 'abData': [12, ['array', lambda x: x.cbData, ['unsigned char']]]}], '_IMAGE_ENTRY_IN_SESSION': [None, {'Link': [0, ['_LIST_ENTRY']], 'Address': [8, ['pointer', ['address']]], 'LastAddress': [12, ['pointer', ['address']]], 'DataTableEntry': [24, ['pointer', ['_LDR_DATA_TABLE_ENTRY']]]}], 'tagEVENTHOOK': [48, {'phkNext': [12, ['pointer', ['tagEVENTHOOK']]], 'eventMin': [16, ['Enumeration', dict(target='unsigned long', choices=consts.EVENT_ID_ENUM)]], 'eventMax': [20, ['Enumeration', dict(target='unsigned long', choices=consts.EVENT_ID_ENUM)]], 'dwFlags': [24, ['unsigned long']], 'idProcess': [28, ['unsigned long']], 'idThread': [32, ['unsigned long']], 'offPfn': [36, ['unsigned long']], 'ihmod': [40, ['long']]}], 'tagHANDLETYPEINFO': [12, {'fnDestroy': [0, ['pointer', ['void']]], 'dwAllocTag': [4, ['String', dict(length=4)]], 'bObjectCreateFlags': [8, ['Flags', {'target': 'unsigned char', 'bitmap': {'OCF_THREADOWNED': 0, 'OCF_PROCESSOWNED': 1, 'OCF_MARKPROCESS': 2, 'OCF_USEPOOLQUOTA': 3, 'OCF_DESKTOPHEAP': 4, 'OCF_USEPOOLIFNODESKTOP': 5, 'OCF_SHAREDHEAP': 6, 'OCF_VARIABLESIZE': 7}}]]}]})

class Win32Kx64VTypes(obj.ProfileModification):
    """Applies to all x64 windows profiles. 

    These are vtypes not included in win32k.sys PDB.
    """
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update({'tagWIN32HEAP': [None, {'Heap': [0, ['_HEAP']]}], '_IMAGE_ENTRY_IN_SESSION': [None, {'Link': [0, ['_LIST_ENTRY']], 'Address': [16, ['pointer', ['void']]], 'LastAddress': [24, ['pointer', ['address']]], 'DataTableEntry': [32, ['pointer', ['_LDR_DATA_TABLE_ENTRY']]]}], 'tagCLIPDATA': [None, {'cbData': [16, ['unsigned int']], 'abData': [20, ['array', lambda x: x.cbData, ['unsigned char']]]}], 'tagEVENTHOOK': [None, {'phkNext': [24, ['pointer', ['tagEVENTHOOK']]], 'eventMin': [32, ['Enumeration', dict(target='unsigned long', choices=consts.EVENT_ID_ENUM)]], 'eventMax': [36, ['Enumeration', dict(target='unsigned long', choices=consts.EVENT_ID_ENUM)]], 'dwFlags': [40, ['unsigned long']], 'idProcess': [44, ['unsigned long']], 'idThread': [48, ['unsigned long']], 'offPfn': [64, ['unsigned long long']], 'ihmod': [72, ['long']]}], 'tagHANDLETYPEINFO': [16, {'fnDestroy': [0, ['pointer', ['void']]], 'dwAllocTag': [8, ['String', dict(length=4)]], 'bObjectCreateFlags': [12, ['Flags', {'target': 'unsigned char', 'bitmap': {'OCF_THREADOWNED': 0, 'OCF_PROCESSOWNED': 1, 'OCF_MARKPROCESS': 2, 'OCF_USEPOOLQUOTA': 3, 'OCF_DESKTOPHEAP': 4, 'OCF_USEPOOLIFNODESKTOP': 5, 'OCF_SHAREDHEAP': 6, 'OCF_VARIABLESIZE': 7}}]]}]})

class XPx86SessionOverlay(obj.ProfileModification):
    """Apply the ResidentProcessCount overlay for x86 XP session spaces"""
    before = ['WindowsOverlay']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 1}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_MM_SESSION_SPACE': [None, {'ResidentProcessCount': [584, ['long']]}]})