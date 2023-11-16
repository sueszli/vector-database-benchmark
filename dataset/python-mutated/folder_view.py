import os
import pickle
import random
import sys
import commctrl
import pythoncom
import win32api
import win32con
import winerror
import winxpgui as win32gui
from win32com.axcontrol import axcontrol
from win32com.propsys import propsys
from win32com.server.exception import COMException
from win32com.server.util import NewEnum as _NewEnum, wrap as _wrap
from win32com.shell import shell, shellcon
from win32com.util import IIDToInterfaceName
GUID = pythoncom.MakeIID
debug = 0

def wrap(ob, iid=None):
    if False:
        i = 10
        return i + 15
    return _wrap(ob, iid, useDispatcher=debug > 0)

def NewEnum(seq, iid):
    if False:
        return 10
    return _NewEnum(seq, iid=iid, useDispatcher=debug > 0)
_sids = {}

def LoadString(sid):
    if False:
        return 10
    return _sids[sid]
_last_ids = 0

def _make_ids(s):
    if False:
        while True:
            i = 10
    global _last_ids
    _last_ids += 1
    _sids[_last_ids] = s
    return _last_ids
IDS_UNSPECIFIED = _make_ids('unspecified')
IDS_SMALL = _make_ids('small')
IDS_MEDIUM = _make_ids('medium')
IDS_LARGE = _make_ids('large')
IDS_CIRCLE = _make_ids('circle')
IDS_TRIANGLE = _make_ids('triangle')
IDS_RECTANGLE = _make_ids('rectangle')
IDS_POLYGON = _make_ids('polygon')
IDS_DISPLAY = _make_ids('Display')
IDS_DISPLAY_TT = _make_ids('Display the item.')
IDS_SETTINGS = _make_ids('Settings')
IDS_SETTING1 = _make_ids('Setting 1')
IDS_SETTING2 = _make_ids('Setting 2')
IDS_SETTING3 = _make_ids('Setting 3')
IDS_SETTINGS_TT = _make_ids('Modify settings.')
IDS_SETTING1_TT = _make_ids('Modify setting 1.')
IDS_SETTING2_TT = _make_ids('Modify setting 2.')
IDS_SETTING3_TT = _make_ids('Modify setting 3.')
IDS_LESSTHAN5 = _make_ids('Less Than 5')
IDS_5ORGREATER = _make_ids('Five or Greater')
del _make_ids, _last_ids
IDI_ICON1 = 100
IDI_SETTINGS = 101
CAT_GUID_NAME = GUID('{de094c9d-c65a-11dc-ba21-005056c00008}')
CAT_GUID_SIZE = GUID('{de094c9e-c65a-11dc-ba21-005056c00008}')
CAT_GUID_SIDES = GUID('{de094c9f-c65a-11dc-ba21-005056c00008}')
CAT_GUID_LEVEL = GUID('{de094ca0-c65a-11dc-ba21-005056c00008}')
CAT_GUID_VALUE = '{de094ca1-c65a-11dc-ba21-005056c00008}'
GUID_Display = GUID('{4d6c2fdd-c689-11dc-ba21-005056c00008}')
GUID_Settings = GUID('{4d6c2fde-c689-11dc-ba21-005056c00008}')
GUID_Setting1 = GUID('{4d6c2fdf-c689-11dc-ba21-005056c00008}')
GUID_Setting2 = GUID('{4d6c2fe0-c689-11dc-ba21-005056c00008}')
GUID_Setting3 = GUID('{4d6c2fe1-c689-11dc-ba21-005056c00008}')
PKEY_ItemNameDisplay = ('{B725F130-47EF-101A-A5F1-02608C9EEBAC}', 10)
PKEY_PropList_PreviewDetails = ('{C9944A21-A406-48FE-8225-AEC7E24C211B}', 8)
PID_SOMETHING = 3
PKEY_Sample_AreaSize = ('{d6f5e341-c65c-11dc-ba21-005056c00008}', PID_SOMETHING)
PKEY_Sample_NumberOfSides = ('{d6f5e342-c65c-11dc-ba21-005056c00008}', PID_SOMETHING)
PKEY_Sample_DirectoryLevel = ('{d6f5e343-c65c-11dc-ba21-005056c00008}', PID_SOMETHING)

def pidl_to_item(pidl):
    if False:
        i = 10
        return i + 15
    return pickle.loads(pidl[-1])

def make_item_enum(level, flags):
    if False:
        return 10
    pidls = []
    nums = 'zero one two three four five size seven eight nine ten'.split()
    for (i, name) in enumerate(nums):
        size = random.randint(0, 255)
        sides = 1
        while sides in [1, 2]:
            sides = random.randint(0, 5)
        is_folder = i % 2 != 0
        skip = False
        if not flags & shellcon.SHCONTF_STORAGE:
            if is_folder:
                skip = not flags & shellcon.SHCONTF_FOLDERS
            else:
                skip = not flags & shellcon.SHCONTF_NONFOLDERS
        if not skip:
            data = {'name': name, 'size': size, 'sides': sides, 'level': level, 'is_folder': is_folder}
            pidls.append([pickle.dumps(data)])
    return NewEnum(pidls, shell.IID_IEnumIDList)

def DisplayItem(shell_item_array, hwnd_parent=0):
    if False:
        i = 10
        return i + 15
    if shell_item_array is None:
        msg = 'You must select something!'
    else:
        si = shell_item_array.GetItemAt(0)
        name = si.GetDisplayName(shellcon.SIGDN_NORMALDISPLAY)
        msg = '%d items selected, first is %r' % (shell_item_array.GetCount(), name)
    win32gui.MessageBox(hwnd_parent, msg, 'Hello', win32con.MB_OK)

class Command:

    def __init__(self, guid, ids, ids_tt, idi, flags, callback, children):
        if False:
            return 10
        self.guid = guid
        self.ids = ids
        self.ids_tt = ids_tt
        self.idi = idi
        self.flags = flags
        self.callback = callback
        self.children = children
        assert not children or isinstance(children[0], Command)

    def tuple(self):
        if False:
            print('Hello World!')
        return (self.guid, self.ids, self.ids_tt, self.idi, self.flags, self.callback, self.children)

def onDisplay(items, bindctx):
    if False:
        while True:
            i = 10
    DisplayItem(items)

def onSetting1(items, bindctx):
    if False:
        return 10
    win32gui.MessageBox(0, LoadString(IDS_SETTING1), 'Hello', win32con.MB_OK)

def onSetting2(items, bindctx):
    if False:
        print('Hello World!')
    win32gui.MessageBox(0, LoadString(IDS_SETTING2), 'Hello', win32con.MB_OK)

def onSetting3(items, bindctx):
    if False:
        return 10
    win32gui.MessageBox(0, LoadString(IDS_SETTING3), 'Hello', win32con.MB_OK)
taskSettings = [Command(GUID_Setting1, IDS_SETTING1, IDS_SETTING1_TT, IDI_SETTINGS, 0, onSetting1, None), Command(GUID_Setting2, IDS_SETTING2, IDS_SETTING2_TT, IDI_SETTINGS, 0, onSetting2, None), Command(GUID_Setting3, IDS_SETTING3, IDS_SETTING3_TT, IDI_SETTINGS, 0, onSetting3, None)]
tasks = [Command(GUID_Display, IDS_DISPLAY, IDS_DISPLAY_TT, IDI_ICON1, 0, onDisplay, None), Command(GUID_Settings, IDS_SETTINGS, IDS_SETTINGS_TT, IDI_SETTINGS, shellcon.ECF_HASSUBCOMMANDS, None, taskSettings)]

class ExplorerCommandProvider:
    _com_interfaces_ = [shell.IID_IExplorerCommandProvider]
    _public_methods_ = shellcon.IExplorerCommandProvider_Methods

    def GetCommands(self, site, iid):
        if False:
            i = 10
            return i + 15
        items = [wrap(ExplorerCommand(t)) for t in tasks]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)

class ExplorerCommand:
    _com_interfaces_ = [shell.IID_IExplorerCommand]
    _public_methods_ = shellcon.IExplorerCommand_Methods

    def __init__(self, cmd):
        if False:
            return 10
        self.cmd = cmd

    def GetTitle(self, pidl):
        if False:
            while True:
                i = 10
        return LoadString(self.cmd.ids)

    def GetToolTip(self, pidl):
        if False:
            while True:
                i = 10
        return LoadString(self.cmd.ids_tt)

    def GetIcon(self, pidl):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetState(self, shell_items, slow_ok):
        if False:
            return 10
        return shellcon.ECS_ENABLED

    def GetFlags(self):
        if False:
            print('Hello World!')
        return self.cmd.flags

    def GetCanonicalName(self):
        if False:
            return 10
        return self.cmd.guid

    def Invoke(self, items, bind_ctx):
        if False:
            return 10
        if self.cmd.callback:
            self.cmd.callback(items, bind_ctx)
        else:
            print('No callback for command ', LoadString(self.cmd.ids))

    def EnumSubCommands(self):
        if False:
            print('Hello World!')
        if not self.cmd.children:
            return None
        items = [wrap(ExplorerCommand(c)) for c in self.cmd.children]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)

class FolderViewCategorizer:
    _com_interfaces_ = [shell.IID_ICategorizer]
    _public_methods_ = shellcon.ICategorizer_Methods
    description = None

    def __init__(self, shell_folder):
        if False:
            for i in range(10):
                print('nop')
        self.sf = shell_folder

    def CompareCategory(self, flags, cat1, cat2):
        if False:
            print('Hello World!')
        return cat1 - cat2

    def GetDescription(self, cch):
        if False:
            i = 10
            return i + 15
        return self.description

    def GetCategoryInfo(self, catid):
        if False:
            i = 10
            return i + 15
        return (0, str(catid))

class FolderViewCategorizer_Name(FolderViewCategorizer):
    description = 'Alphabetical'

    def GetCategory(self, pidls):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            ret.append(val)
        return ret

class FolderViewCategorizer_Size(FolderViewCategorizer):
    description = 'Group By Size'

    def GetCategory(self, pidls):
        if False:
            return 10
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_Sample_AreaSize)
            val = int(val)
            if val < 255 // 3:
                cid = IDS_SMALL
            elif val < 2 * 255 // 3:
                cid = IDS_MEDIUM
            else:
                cid = IDS_LARGE
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        if False:
            while True:
                i = 10
        return (0, LoadString(catid))

class FolderViewCategorizer_Sides(FolderViewCategorizer):
    description = 'Group By Sides'

    def GetCategory(self, pidls):
        if False:
            return 10
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val == 0:
                cid = IDS_CIRCLE
            elif val == 3:
                cid = IDS_TRIANGLE
            elif val == 4:
                cid = IDS_RECTANGLE
            elif val == 5:
                cid = IDS_POLYGON
            else:
                cid = IDS_UNSPECIFIED
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        if False:
            while True:
                i = 10
        return (0, LoadString(catid))

class FolderViewCategorizer_Value(FolderViewCategorizer):
    description = 'Group By Value'

    def GetCategory(self, pidls):
        if False:
            i = 10
            return i + 15
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val in 'one two three four'.split():
                ret.append(IDS_LESSTHAN5)
            else:
                ret.append(IDS_5ORGREATER)
        return ret

    def GetCategoryInfo(self, catid):
        if False:
            i = 10
            return i + 15
        return (0, LoadString(catid))

class FolderViewCategorizer_Level(FolderViewCategorizer):
    description = 'Group By Value'

    def GetCategory(self, pidls):
        if False:
            i = 10
            return i + 15
        return [self.sf.GetDetailsEx(pidl, PKEY_Sample_DirectoryLevel) for pidl in pidls]

class ViewCategoryProvider:
    _com_interfaces_ = [shell.IID_ICategoryProvider]
    _public_methods_ = shellcon.ICategoryProvider_Methods

    def __init__(self, shell_folder):
        if False:
            i = 10
            return i + 15
        self.shell_folder = shell_folder

    def CanCategorizeOnSCID(self, pkey):
        if False:
            for i in range(10):
                print('nop')
        return pkey in [PKEY_ItemNameDisplay, PKEY_Sample_AreaSize, PKEY_Sample_NumberOfSides, PKEY_Sample_DirectoryLevel]

    def CreateCategory(self, guid, iid):
        if False:
            print('Hello World!')
        if iid == shell.IID_ICategorizer:
            if guid == CAT_GUID_NAME:
                klass = FolderViewCategorizer_Name
            elif guid == CAT_GUID_SIDES:
                klass = FolderViewCategorizer_Sides
            elif guid == CAT_GUID_SIZE:
                klass = FolderViewCategorizer_Size
            elif guid == CAT_GUID_VALUE:
                klass = FolderViewCategorizer_Value
            elif guid == CAT_GUID_LEVEL:
                klass = FolderViewCategorizer_Level
            else:
                raise COMException(hresult=winerror.E_INVALIDARG)
            return wrap(klass(self.shell_folder))
        raise COMException(hresult=winerror.E_NOINTERFACE)

    def EnumCategories(self):
        if False:
            return 10
        seq = [CAT_GUID_VALUE]
        return NewEnum(seq, pythoncom.IID_IEnumGUID)

    def GetCategoryForSCID(self, scid):
        if False:
            print('Hello World!')
        if scid == PKEY_ItemNameDisplay:
            guid = CAT_GUID_NAME
        elif scid == PKEY_Sample_AreaSize:
            guid = CAT_GUID_SIZE
        elif scid == PKEY_Sample_NumberOfSides:
            guid = CAT_GUID_SIDES
        elif scid == PKEY_Sample_DirectoryLevel:
            guid = CAT_GUID_LEVEL
        elif scid == pythoncom.IID_NULL:
            guid = CAT_GUID_VALUE
        else:
            raise COMException(hresult=winerror.E_INVALIDARG)
        return guid

    def GetCategoryName(self, guid, cch):
        if False:
            return 10
        if guid == CAT_GUID_VALUE:
            return 'Value'
        raise COMException(hresult=winerror.E_FAIL)

    def GetDefaultCategory(self):
        if False:
            print('Hello World!')
        return (CAT_GUID_LEVEL, (pythoncom.IID_NULL, 0))
MENUVERB_DISPLAY = 0
folderViewImplContextMenuIDs = [('display', MENUVERB_DISPLAY, 0)]

class ContextMenu:
    _reg_progid_ = 'Python.ShellFolderSample.ContextMenu'
    _reg_desc_ = 'Python FolderView Context Menu'
    _reg_clsid_ = '{fed40039-021f-4011-87c5-6188b9979764}'
    _com_interfaces_ = [shell.IID_IShellExtInit, shell.IID_IContextMenu, axcontrol.IID_IObjectWithSite]
    _public_methods_ = shellcon.IContextMenu_Methods + shellcon.IShellExtInit_Methods + ['GetSite', 'SetSite']
    _context_menu_type_ = 'PythonFolderViewSampleType'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.site = None
        self.dataobj = None

    def Initialize(self, folder, dataobj, hkey):
        if False:
            print('Hello World!')
        self.dataobj = dataobj

    def QueryContextMenu(self, hMenu, indexMenu, idCmdFirst, idCmdLast, uFlags):
        if False:
            while True:
                i = 10
        s = LoadString(IDS_DISPLAY)
        win32gui.InsertMenu(hMenu, indexMenu, win32con.MF_BYPOSITION, idCmdFirst + MENUVERB_DISPLAY, s)
        indexMenu += 1
        return 1

    def InvokeCommand(self, ci):
        if False:
            i = 10
            return i + 15
        (mask, hwnd, verb, params, dir, nShow, hotkey, hicon) = ci
        for (verb_name, verb_id, flag) in folderViewImplContextMenuIDs:
            if isinstance(verb, int):
                matches = verb == verb_id
            else:
                matches = verb == verb_name
            if matches:
                break
        else:
            assert False, ci
        if verb_id == MENUVERB_DISPLAY:
            sia = shell.SHCreateShellItemArrayFromDataObject(self.dataobj)
            DisplayItem(hwnd, sia)
        else:
            assert False, ci

    def GetCommandString(self, cmd, typ):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetSite(self, site):
        if False:
            print('Hello World!')
        self.site = site

    def GetSite(self, iid):
        if False:
            i = 10
            return i + 15
        return self.site

class ShellFolder:
    _com_interfaces_ = [shell.IID_IBrowserFrameOptions, pythoncom.IID_IPersist, shell.IID_IPersistFolder, shell.IID_IPersistFolder2, shell.IID_IShellFolder, shell.IID_IShellFolder2]
    _public_methods_ = shellcon.IBrowserFrame_Methods + shellcon.IPersistFolder2_Methods + shellcon.IShellFolder2_Methods
    _reg_progid_ = 'Python.ShellFolderSample.Folder2'
    _reg_desc_ = 'Python FolderView sample'
    _reg_clsid_ = '{bb8c24ad-6aaa-4cec-ac5e-c429d5f57627}'
    max_levels = 5

    def __init__(self, level=0):
        if False:
            for i in range(10):
                print('nop')
        self.current_level = level
        self.pidl = None

    def ParseDisplayName(self, hwnd, reserved, displayName, attr):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumObjects(self, hwndOwner, flags):
        if False:
            for i in range(10):
                print('nop')
        if self.current_level >= self.max_levels:
            return None
        return make_item_enum(self.current_level + 1, flags)

    def BindToObject(self, pidl, bc, iid):
        if False:
            print('Hello World!')
        tail = pidl_to_item(pidl)
        if iid not in ShellFolder._com_interfaces_:
            raise COMException(hresult=winerror.E_NOTIMPL)
        child = ShellFolder(self.current_level + 1)
        child.Initialize(self.pidl + pidl)
        return wrap(child, iid)

    def BindToStorage(self, pidl, bc, iid):
        if False:
            print('Hello World!')
        return self.BindToObject(pidl, bc, iid)

    def CompareIDs(self, param, id1, id2):
        if False:
            return 10
        return 0

    def CreateViewObject(self, hwnd, iid):
        if False:
            return 10
        if iid == shell.IID_IShellView:
            com_folder = wrap(self)
            return shell.SHCreateShellFolderView(com_folder)
        elif iid == shell.IID_ICategoryProvider:
            return wrap(ViewCategoryProvider(self))
        elif iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwnd, None, self.pidl, ws, None)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExplorerCommandProvider:
            return wrap(ExplorerCommandProvider())
        else:
            raise COMException(hresult=winerror.E_NOINTERFACE)

    def GetAttributesOf(self, pidls, attrFlags):
        if False:
            i = 10
            return i + 15
        assert len(pidls) == 1, 'sample only expects 1 too!'
        assert len(pidls[0]) == 1, 'expect relative pidls!'
        item = pidl_to_item(pidls[0])
        flags = 0
        if item['is_folder']:
            flags |= shellcon.SFGAO_FOLDER
        if item['level'] < self.max_levels:
            flags |= shellcon.SFGAO_HASSUBFOLDER
        return flags

    def GetUIObjectOf(self, hwndOwner, pidls, iid, inout):
        if False:
            print('Hello World!')
        assert len(pidls) == 1, 'oops - arent expecting more than one!'
        assert len(pidls[0]) == 1, 'assuming relative pidls!'
        item = pidl_to_item(pidls[0])
        if iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwndOwner, None, self.pidl, ws, pidls)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExtractIconW:
            dxi = shell.SHCreateDefaultExtractIcon()
            if item['is_folder']:
                dxi.SetNormalIcon('shell32.dll', 4)
            else:
                dxi.SetNormalIcon('shell32.dll', 1)
            return dxi
        elif iid == pythoncom.IID_IDataObject:
            return shell.SHCreateDataObject(self.pidl, pidls, None, iid)
        elif iid == shell.IID_IQueryAssociations:
            elts = []
            if item['is_folder']:
                elts.append((shellcon.ASSOCCLASS_FOLDER, None, None))
            elts.append((shellcon.ASSOCCLASS_PROGID_STR, None, ContextMenu._context_menu_type_))
            return shell.AssocCreateForClasses(elts, iid)
        raise COMException(hresult=winerror.E_NOINTERFACE)

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            return 10
        item = pidl_to_item(pidl)
        if flags & shellcon.SHGDN_FORPARSING:
            if flags & shellcon.SHGDN_INFOLDER:
                return item['name']
            else:
                if flags & shellcon.SHGDN_FORADDRESSBAR:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEEDITING
                else:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEPARSING
                parent = shell.SHGetNameFromIDList(self.pidl, sigdn)
                return parent + '\\' + item['name']
        else:
            return item['name']

    def SetNameOf(self, hwndOwner, pidl, new_name, flags):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetClassID(self):
        if False:
            for i in range(10):
                print('nop')
        return self._reg_clsid_

    def Initialize(self, pidl):
        if False:
            return 10
        self.pidl = pidl

    def EnumSearches(self):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOINTERFACE)

    def GetDefaultColumn(self, dwres):
        if False:
            return 10
        return (0, 0)

    def GetDefaultColumnState(self, iCol):
        if False:
            return 10
        if iCol < 3:
            return shellcon.SHCOLSTATE_ONBYDEFAULT | shellcon.SHCOLSTATE_TYPE_STR
        raise COMException(hresult=winerror.E_INVALIDARG)

    def GetDefaultSearchGUID(self):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def _GetColumnDisplayName(self, pidl, pkey):
        if False:
            print('Hello World!')
        item = pidl_to_item(pidl)
        is_folder = item['is_folder']
        if pkey == PKEY_ItemNameDisplay:
            val = item['name']
        elif pkey == PKEY_Sample_AreaSize and (not is_folder):
            val = '%d Sq. Ft.' % item['size']
        elif pkey == PKEY_Sample_NumberOfSides and (not is_folder):
            val = str(item['sides'])
        elif pkey == PKEY_Sample_DirectoryLevel:
            val = str(item['level'])
        else:
            val = ''
        return val

    def GetDetailsEx(self, pidl, pkey):
        if False:
            print('Hello World!')
        item = pidl_to_item(pidl)
        is_folder = item['is_folder']
        if not is_folder and pkey == PKEY_PropList_PreviewDetails:
            return 'prop:Sample.AreaSize;Sample.NumberOfSides;Sample.DirectoryLevel'
        return self._GetColumnDisplayName(pidl, pkey)

    def GetDetailsOf(self, pidl, iCol):
        if False:
            for i in range(10):
                print('nop')
        key = self.MapColumnToSCID(iCol)
        if pidl is None:
            data = [(commctrl.LVCFMT_LEFT, 'Name'), (commctrl.LVCFMT_CENTER, 'Size'), (commctrl.LVCFMT_CENTER, 'Sides'), (commctrl.LVCFMT_CENTER, 'Level')]
            if iCol >= len(data):
                raise COMException(hresult=winerror.E_FAIL)
            (fmt, val) = data[iCol]
        else:
            fmt = 0
            val = self._GetColumnDisplayName(pidl, key)
        cxChar = 24
        return (fmt, cxChar, val)

    def MapColumnToSCID(self, iCol):
        if False:
            i = 10
            return i + 15
        data = [PKEY_ItemNameDisplay, PKEY_Sample_AreaSize, PKEY_Sample_NumberOfSides, PKEY_Sample_DirectoryLevel]
        if iCol >= len(data):
            raise COMException(hresult=winerror.E_FAIL)
        return data[iCol]

    def GetCurFolder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pidl

def get_schema_fname():
    if False:
        i = 10
        return i + 15
    me = win32api.GetFullPathName(__file__)
    sc = os.path.splitext(me)[0] + '.propdesc'
    assert os.path.isfile(sc), sc
    return sc

def DllRegisterServer():
    if False:
        print('Hello World!')
    import winreg
    if sys.getwindowsversion()[0] < 6:
        print('This sample only works on Vista')
        sys.exit(1)
    key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\' + ShellFolder._reg_clsid_)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellFolder._reg_desc_)
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 'CLSID\\' + ShellFolder._reg_clsid_ + '\\ShellFolder')
    attr = shellcon.SFGAO_FOLDER | shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_BROWSABLE
    import struct
    s = struct.pack('i', attr)
    winreg.SetValueEx(key, 'Attributes', 0, winreg.REG_BINARY, s)
    keypath = '{}\\shellex\\ContextMenuHandlers\\{}'.format(ContextMenu._context_menu_type_, ContextMenu._reg_desc_)
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, keypath)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ContextMenu._reg_clsid_)
    propsys.PSRegisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, 'registration complete.')

def DllUnregisterServer():
    if False:
        for i in range(10):
            print('nop')
    import winreg
    paths = ['SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\' + ShellFolder._reg_clsid_, '{}\\shellex\\ContextMenuHandlers\\{}'.format(ContextMenu._context_menu_type_, ContextMenu._reg_desc_)]
    for path in paths:
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, path)
        except OSError as details:
            import errno
            if details.errno != errno.ENOENT:
                print(f'FAILED to remove {path}: {details}')
    propsys.PSUnregisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(ShellFolder, ContextMenu, debug=debug, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)