import _thread
import os
import pyclbr
import sys
import commctrl
import pythoncom
import win32api
import win32con
import win32gui
import win32gui_struct
import winerror
from pywin.scintilla import scintillacon
from win32com.server.exception import COMException
from win32com.server.util import NewEnum, wrap
from win32com.shell import shell, shellcon
from win32com.util import IIDToInterfaceName
debug = 0
if debug:
    import win32traceutil
com_auto_reload = True

def GetFolderAndPIDLForPath(filename):
    if False:
        print('Hello World!')
    desktop = shell.SHGetDesktopFolder()
    info = desktop.ParseDisplayName(0, None, os.path.abspath(filename))
    (cchEaten, pidl, attr) = info
    folder = desktop
    while len(pidl) > 1:
        this = pidl.pop(0)
        folder = folder.BindToObject([this], None, shell.IID_IShellFolder)
    return (folder, pidl)
clbr_modules = {}

def get_clbr_for_file(path):
    if False:
        for i in range(10):
            print('nop')
    try:
        objects = clbr_modules[path]
    except KeyError:
        (dir, filename) = os.path.split(path)
        (base, ext) = os.path.splitext(filename)
        objects = pyclbr.readmodule_ex(base, [dir])
        clbr_modules[path] = objects
    return objects

class ShellFolderBase:
    _com_interfaces_ = [shell.IID_IBrowserFrameOptions, pythoncom.IID_IPersist, shell.IID_IPersistFolder, shell.IID_IShellFolder]
    _public_methods_ = shellcon.IBrowserFrame_Methods + shellcon.IPersistFolder_Methods + shellcon.IShellFolder_Methods

    def GetFrameOptions(self, mask):
        if False:
            while True:
                i = 10
        return 0

    def ParseDisplayName(self, hwnd, reserved, displayName, attr):
        if False:
            for i in range(10):
                print('nop')
        print('ParseDisplayName', displayName)

    def BindToStorage(self, pidl, bc, iid):
        if False:
            i = 10
            return i + 15
        print('BTS', iid, IIDToInterfaceName(iid))

    def BindToObject(self, pidl, bc, iid):
        if False:
            for i in range(10):
                print('nop')
        final_pidl = pidl[-1]
        (typ, extra) = final_pidl.split('\x00', 1)
        if typ == 'directory':
            klass = ShellFolderDirectory
        elif typ == 'file':
            klass = ShellFolderFile
        elif typ == 'object':
            klass = ShellFolderObject
        else:
            raise RuntimeError('What is ' + repr(typ))
        ret = wrap(klass(extra), iid, useDispatcher=debug > 0)
        return ret

class ShellFolderFileSystem(ShellFolderBase):

    def _GetFolderAndPIDLForPIDL(self, my_idl):
        if False:
            print('Hello World!')
        (typ, name) = my_idl[0].split('\x00')
        return GetFolderAndPIDLForPath(name)

    def CompareIDs(self, param, id1, id2):
        if False:
            return 10
        if id1 < id2:
            return -1
        if id1 == id2:
            return 0
        return 1

    def GetUIObjectOf(self, hwndOwner, pidls, iid, inout):
        if False:
            for i in range(10):
                print('nop')
        assert len(pidls) == 1, 'oops - arent expecting more than one!'
        pidl = pidls[0]
        (folder, child_pidl) = self._GetFolderAndPIDLForPIDL(pidl)
        try:
            (inout, ret) = folder.GetUIObjectOf(hwndOwner, [child_pidl], iid, inout, iid)
        except pythoncom.com_error as exc:
            raise COMException(hresult=exc.hresult)
        return (inout, ret)

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            return 10
        (folder, child_pidl) = self._GetFolderAndPIDLForPIDL(pidl)
        ret = folder.GetDisplayNameOf(child_pidl, flags)
        return ret

    def GetAttributesOf(self, pidls, attrFlags):
        if False:
            print('Hello World!')
        ret_flags = -1
        for pidl in pidls:
            pidl = pidl[0]
            (typ, name) = pidl.split('\x00')
            flags = shellcon.SHGFI_ATTRIBUTES
            (rc, info) = shell.SHGetFileInfo(name, 0, flags)
            (hIcon, iIcon, dwAttr, name, typeName) = info
            extras = shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_FOLDER | shellcon.SFGAO_FILESYSANCESTOR | shellcon.SFGAO_BROWSABLE
            ret_flags &= dwAttr | extras
        return ret_flags

class ShellFolderDirectory(ShellFolderFileSystem):

    def __init__(self, path):
        if False:
            while True:
                i = 10
        self.path = os.path.abspath(path)

    def CreateViewObject(self, hwnd, iid):
        if False:
            for i in range(10):
                print('nop')
        (folder, child_pidl) = GetFolderAndPIDLForPath(self.path)
        return folder.CreateViewObject(hwnd, iid)

    def EnumObjects(self, hwndOwner, flags):
        if False:
            i = 10
            return i + 15
        pidls = []
        for fname in os.listdir(self.path):
            fqn = os.path.join(self.path, fname)
            if os.path.isdir(fqn):
                type_name = 'directory'
                type_class = ShellFolderDirectory
            else:
                (base, ext) = os.path.splitext(fname)
                if ext in ['.py', '.pyw']:
                    type_class = ShellFolderFile
                    type_name = 'file'
                else:
                    type_class = None
            if type_class is not None:
                pidls.append([type_name + '\x00' + fqn])
        return NewEnum(pidls, iid=shell.IID_IEnumIDList, useDispatcher=debug > 0)

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            i = 10
            return i + 15
        final_pidl = pidl[-1]
        full_fname = final_pidl.split('\x00')[-1]
        return os.path.split(full_fname)[-1]

    def GetAttributesOf(self, pidls, attrFlags):
        if False:
            return 10
        return shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_FOLDER | shellcon.SFGAO_FILESYSANCESTOR | shellcon.SFGAO_BROWSABLE

class ShellFolderFile(ShellFolderBase):

    def __init__(self, path):
        if False:
            return 10
        self.path = os.path.abspath(path)

    def EnumObjects(self, hwndOwner, flags):
        if False:
            for i in range(10):
                print('nop')
        objects = get_clbr_for_file(self.path)
        pidls = []
        for (name, ob) in objects.items():
            pidls.append(['object\x00' + self.path + '\x00' + name])
        return NewEnum(pidls, iid=shell.IID_IEnumIDList, useDispatcher=debug > 0)

    def GetAttributesOf(self, pidls, attrFlags):
        if False:
            while True:
                i = 10
        ret_flags = -1
        for pidl in pidls:
            assert len(pidl) == 1, 'Expecting relative pidls'
            pidl = pidl[0]
            (typ, filename, obname) = pidl.split('\x00')
            obs = get_clbr_for_file(filename)
            ob = obs[obname]
            flags = shellcon.SFGAO_BROWSABLE | shellcon.SFGAO_FOLDER | shellcon.SFGAO_FILESYSANCESTOR
            if hasattr(ob, 'methods'):
                flags |= shellcon.SFGAO_HASSUBFOLDER
            ret_flags &= flags
        return ret_flags

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            for i in range(10):
                print('nop')
        assert len(pidl) == 1, 'Expecting relative PIDL'
        (typ, fname, obname) = pidl[0].split('\x00')
        fqname = os.path.splitext(fname)[0] + '.' + obname
        if flags & shellcon.SHGDN_INFOLDER:
            ret = obname
        else:
            ret = fqname
        return ret

    def CreateViewObject(self, hwnd, iid):
        if False:
            return 10
        return wrap(ScintillaShellView(hwnd, self.path), iid, useDispatcher=debug > 0)

class ShellFolderObject(ShellFolderBase):

    def __init__(self, details):
        if False:
            i = 10
            return i + 15
        (self.path, details) = details.split('\x00')
        if details.find('.') > 0:
            (self.class_name, self.method_name) = details.split('.')
        else:
            self.class_name = details
            self.method_name = None

    def CreateViewObject(self, hwnd, iid):
        if False:
            print('Hello World!')
        mod_objects = get_clbr_for_file(self.path)
        object = mod_objects[self.class_name]
        if self.method_name is None:
            lineno = object.lineno
        else:
            lineno = object.methods[self.method_name]
            return wrap(ScintillaShellView(hwnd, self.path, lineno), iid, useDispatcher=debug > 0)

    def EnumObjects(self, hwndOwner, flags):
        if False:
            i = 10
            return i + 15
        assert self.method_name is None, 'Should not be enuming methods!'
        mod_objects = get_clbr_for_file(self.path)
        my_objects = mod_objects[self.class_name]
        pidls = []
        for (func_name, lineno) in my_objects.methods.items():
            pidl = ['object\x00' + self.path + '\x00' + self.class_name + '.' + func_name]
            pidls.append(pidl)
        return NewEnum(pidls, iid=shell.IID_IEnumIDList, useDispatcher=debug > 0)

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            while True:
                i = 10
        assert len(pidl) == 1, 'Expecting relative PIDL'
        (typ, fname, obname) = pidl[0].split('\x00')
        (class_name, method_name) = obname.split('.')
        fqname = os.path.splitext(fname)[0] + '.' + obname
        if flags & shellcon.SHGDN_INFOLDER:
            ret = method_name
        else:
            ret = fqname
        return ret

    def GetAttributesOf(self, pidls, attrFlags):
        if False:
            return 10
        ret_flags = -1
        for pidl in pidls:
            assert len(pidl) == 1, 'Expecting relative pidls'
            flags = shellcon.SFGAO_BROWSABLE | shellcon.SFGAO_FOLDER | shellcon.SFGAO_FILESYSANCESTOR
            ret_flags &= flags
        return ret_flags

class ShellFolderRoot(ShellFolderFileSystem):
    _reg_progid_ = 'Python.ShellExtension.Folder'
    _reg_desc_ = 'Python Path Shell Browser'
    _reg_clsid_ = '{f6287035-3074-4cb5-a8a6-d3c80e206944}'

    def GetClassID(self):
        if False:
            print('Hello World!')
        return self._reg_clsid_

    def Initialize(self, pidl):
        if False:
            return 10
        self.pidl = pidl

    def CreateViewObject(self, hwnd, iid):
        if False:
            while True:
                i = 10
        return wrap(FileSystemView(self, hwnd), iid, useDispatcher=debug > 0)

    def EnumObjects(self, hwndOwner, flags):
        if False:
            i = 10
            return i + 15
        items = [['directory\x00' + p] for p in sys.path if os.path.isdir(p)]
        return NewEnum(items, iid=shell.IID_IEnumIDList, useDispatcher=debug > 0)

    def GetDisplayNameOf(self, pidl, flags):
        if False:
            for i in range(10):
                print('nop')
        final_pidl = pidl[-1]
        display_name = final_pidl.split('\x00')[-1]
        return display_name

class FileSystemView:
    _public_methods_ = shellcon.IShellView_Methods
    _com_interfaces_ = [pythoncom.IID_IOleWindow, shell.IID_IShellView]

    def __init__(self, folder, hwnd):
        if False:
            i = 10
            return i + 15
        self.hwnd_parent = hwnd
        self.hwnd = None
        self.hwnd_child = None
        self.activate_state = None
        self.hmenu = None
        self.browser = None
        self.folder = folder
        self.children = None

    def GetWindow(self):
        if False:
            while True:
                i = 10
        return self.hwnd

    def ContextSensitiveHelp(self, enter_mode):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def CreateViewWindow(self, prev, settings, browser, rect):
        if False:
            while True:
                i = 10
        print('FileSystemView.CreateViewWindow', prev, settings, browser, rect)
        self.cur_foldersettings = settings
        self.browser = browser
        self._CreateMainWindow(prev, settings, browser, rect)
        self._CreateChildWindow(prev)
        browser_ad = win32gui.SendMessage(self.hwnd_parent, win32con.WM_USER + 7, 0, 0)
        browser_ob = pythoncom.ObjectFromAddress(browser_ad, shell.IID_IShellBrowser)
        assert browser == browser_ob
        assert browser.QueryActiveShellView() == browser_ob.QueryActiveShellView()

    def _CreateMainWindow(self, prev, settings, browser, rect):
        if False:
            for i in range(10):
                print('nop')
        style = win32con.WS_CHILD | win32con.WS_VISIBLE
        wclass_name = 'ShellViewDemo_DefView'
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32gui.dllhandle
        wc.lpszClassName = wclass_name
        wc.style = win32con.CS_VREDRAW | win32con.CS_HREDRAW
        try:
            win32gui.RegisterClass(wc)
        except win32gui.error as details:
            if details[0] != winerror.ERROR_CLASS_ALREADY_EXISTS:
                raise
        message_map = {win32con.WM_DESTROY: self.OnDestroy, win32con.WM_COMMAND: self.OnCommand, win32con.WM_NOTIFY: self.OnNotify, win32con.WM_CONTEXTMENU: self.OnContextMenu, win32con.WM_SIZE: self.OnSize}
        self.hwnd = win32gui.CreateWindow(wclass_name, '', style, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], self.hwnd_parent, 0, win32gui.dllhandle, None)
        win32gui.SetWindowLong(self.hwnd, win32con.GWL_WNDPROC, message_map)
        print("View 's hwnd is", self.hwnd)
        return self.hwnd

    def _CreateChildWindow(self, prev):
        if False:
            while True:
                i = 10
        assert self.hwnd_child is None, 'already have a window'
        assert self.cur_foldersettings is not None, 'no settings'
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_BORDER | commctrl.LVS_SHAREIMAGELISTS | commctrl.LVS_EDITLABELS
        (view_mode, view_flags) = self.cur_foldersettings
        if view_mode == shellcon.FVM_ICON:
            style |= commctrl.LVS_ICON | commctrl.LVS_AUTOARRANGE
        elif view_mode == shellcon.FVM_SMALLICON:
            style |= commctrl.LVS_SMALLICON | commctrl.LVS_AUTOARRANGE
        elif view_mode == shellcon.FVM_LIST:
            style |= commctrl.LVS_LIST | commctrl.LVS_AUTOARRANGE
        elif view_mode == shellcon.FVM_DETAILS:
            style |= commctrl.LVS_REPORT | commctrl.LVS_AUTOARRANGE
        else:
            view_mode = shellcon.FVM_DETAILS
            style |= commctrl.LVS_REPORT | commctrl.LVS_AUTOARRANGE
        for (f_flag, l_flag) in [(shellcon.FWF_SINGLESEL, commctrl.LVS_SINGLESEL), (shellcon.FWF_ALIGNLEFT, commctrl.LVS_ALIGNLEFT), (shellcon.FWF_SHOWSELALWAYS, commctrl.LVS_SHOWSELALWAYS)]:
            if view_flags & f_flag:
                style |= l_flag
        self.hwnd_child = win32gui.CreateWindowEx(win32con.WS_EX_CLIENTEDGE, 'SysListView32', None, style, 0, 0, 0, 0, self.hwnd, 1000, 0, None)
        cr = win32gui.GetClientRect(self.hwnd)
        win32gui.MoveWindow(self.hwnd_child, 0, 0, cr[2] - cr[0], cr[3] - cr[1], True)
        (lvc, extras) = win32gui_struct.PackLVCOLUMN(fmt=commctrl.LVCFMT_LEFT, subItem=1, text='Name', cx=300)
        win32gui.SendMessage(self.hwnd_child, commctrl.LVM_INSERTCOLUMN, 0, lvc)
        (lvc, extras) = win32gui_struct.PackLVCOLUMN(fmt=commctrl.LVCFMT_RIGHT, subItem=1, text='Exists', cx=50)
        win32gui.SendMessage(self.hwnd_child, commctrl.LVM_INSERTCOLUMN, 1, lvc)
        self.Refresh()

    def GetCurrentInfo(self):
        if False:
            return 10
        return self.cur_foldersettings

    def UIActivate(self, activate_state):
        if False:
            return 10
        print('OnActivate')

    def _OnActivate(self, activate_state):
        if False:
            return 10
        if self.activate_state == activate_state:
            return
        self._OnDeactivate()
        if activate_state != shellcon.SVUIA_DEACTIVATE:
            assert self.hmenu is None, 'Should have destroyed it!'
            self.hmenu = win32gui.CreateMenu()
            widths = (0, 0, 0, 0, 0, 0)
            self.browser.InsertMenusSB(self.hmenu, widths)
            self._MergeMenus(activate_state)
            self.browser.SetMenuSB(self.hmenu, 0, self.hwnd)
        self.activate_state = activate_state

    def _OnDeactivate(self):
        if False:
            i = 10
            return i + 15
        if self.browser is not None and self.hmenu is not None:
            self.browser.SetMenuSB(0, 0, 0)
            self.browser.RemoveMenusSB(self.hmenu)
            win32gui.DestroyMenu(self.hmenu)
            self.hmenu = None
        self.hsubmenus = None
        self.activate_state = shellcon.SVUIA_DEACTIVATE

    def _MergeMenus(self, activate_state):
        if False:
            i = 10
            return i + 15
        have_sel = activate_state == shellcon.SVUIA_ACTIVATE_FOCUS
        mid = shellcon.FCIDM_MENU_FILE
        (buf, extras) = win32gui_struct.EmptyMENUITEMINFO(win32con.MIIM_SUBMENU)
        win32gui.GetMenuItemInfo(self.hmenu, mid, False, buf)
        data = win32gui_struct.UnpackMENUITEMINFO(buf)
        submenu = data[3]
        print('Do someting with the file menu!')

    def Refresh(self):
        if False:
            while True:
                i = 10
        stateMask = commctrl.LVIS_SELECTED | commctrl.LVIS_DROPHILITED
        state = 0
        self.children = []
        for cid in self.folder.EnumObjects(self.hwnd, 0):
            self.children.append(cid)
        for (row_index, data) in enumerate(self.children):
            assert len(data) == 1, 'expecting just a child PIDL'
            (typ, path) = data[0].split('\x00')
            desc = os.path.exists(path) and 'Yes' or 'No'
            prop_vals = (path, desc)
            (data, extras) = win32gui_struct.PackLVITEM(item=row_index, subItem=0, text=prop_vals[0], state=state, stateMask=stateMask)
            win32gui.SendMessage(self.hwnd_child, commctrl.LVM_INSERTITEM, row_index, data)
            col_index = 1
            for prop_val in prop_vals[1:]:
                (data, extras) = win32gui_struct.PackLVITEM(item=row_index, subItem=col_index, text=prop_val)
                win32gui.SendMessage(self.hwnd_child, commctrl.LVM_SETITEM, 0, data)
                col_index += 1

    def SelectItem(self, pidl, flag):
        if False:
            return 10
        print('Please implement SelectItem for PIDL', pidl)

    def GetItemObject(self, item_num, iid):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TranslateAccelerator(self, msg):
        if False:
            for i in range(10):
                print('nop')
        return winerror.S_FALSE

    def DestroyViewWindow(self):
        if False:
            while True:
                i = 10
        win32gui.DestroyWindow(self.hwnd)
        self.hwnd = None
        print('Destroyed view window')

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        if False:
            print('Hello World!')
        print('OnDestory')

    def OnCommand(self, hwnd, msg, wparam, lparam):
        if False:
            print('Hello World!')
        print('OnCommand')

    def OnNotify(self, hwnd, msg, wparam, lparam):
        if False:
            for i in range(10):
                print('nop')
        (hwndFrom, idFrom, code) = win32gui_struct.UnpackWMNOTIFY(lparam)
        if code == commctrl.NM_SETFOCUS:
            if self.browser is not None:
                self.browser.OnViewWindowActive(None)
            self._OnActivate(shellcon.SVUIA_ACTIVATE_FOCUS)
        elif code == commctrl.NM_KILLFOCUS:
            self._OnDeactivate()
        elif code == commctrl.NM_DBLCLK:
            sel = []
            n = -1
            while 1:
                n = win32gui.SendMessage(self.hwnd_child, commctrl.LVM_GETNEXTITEM, n, commctrl.LVNI_SELECTED)
                if n == -1:
                    break
                sel.append(self.children[n][-1:])
            print('Selection is', sel)
            hmenu = win32gui.CreateMenu()
            try:
                (inout, cm) = self.folder.GetUIObjectOf(self.hwnd_parent, sel, shell.IID_IContextMenu, 0)
                flags = shellcon.CMF_DEFAULTONLY
                try:
                    self.browser.GetControlWindow(shellcon.FCW_TREE)
                    flags |= shellcon.CMF_EXPLORE
                except pythoncom.com_error:
                    pass
                if 0:
                    id_cmd_first = 1
                    cm.QueryContextMenu(hmenu, 0, id_cmd_first, -1, flags)
                    cmd = win32gui.GetMenuDefaultItem(hmenu, False, 0)
                    if cmd == -1:
                        print('Oops: _doDefaultActionFor found no default menu')
                    else:
                        ci = (0, self.hwnd_parent, cmd - id_cmd_first, None, None, 0, 0, 0)
                        cm.InvokeCommand(ci)
                else:
                    rv = shell.ShellExecuteEx(hwnd=self.hwnd_parent, nShow=win32con.SW_NORMAL, lpClass='folder', lpVerb='explore', lpIDList=sel[0])
                    print('ShellExecuteEx returned', rv)
            finally:
                win32gui.DestroyMenu(hmenu)

    def OnContextMenu(self, hwnd, msg, wparam, lparam):
        if False:
            return 10
        pidls = []
        n = -1
        while 1:
            n = win32gui.SendMessage(self.hwnd_child, commctrl.LVM_GETNEXTITEM, n, commctrl.LVNI_SELECTED)
            if n == -1:
                break
            pidls.append(self.children[n][-1:])
        spt = win32api.GetCursorPos()
        if not pidls:
            print('Ignoring background click')
            return
        (inout, cm) = self.folder.GetUIObjectOf(self.hwnd_parent, pidls, shell.IID_IContextMenu, 0)
        hmenu = win32gui.CreatePopupMenu()
        sel = None
        try:
            flags = 0
            try:
                self.browser.GetControlWindow(shellcon.FCW_TREE)
                flags |= shellcon.CMF_EXPLORE
            except pythoncom.com_error:
                pass
            id_cmd_first = 1
            cm.QueryContextMenu(hmenu, 0, id_cmd_first, -1, flags)
            tpm_flags = win32con.TPM_LEFTALIGN | win32con.TPM_RETURNCMD | win32con.TPM_RIGHTBUTTON
            sel = win32gui.TrackPopupMenu(hmenu, tpm_flags, spt[0], spt[1], 0, self.hwnd, None)
            print('TrackPopupMenu returned', sel)
        finally:
            win32gui.DestroyMenu(hmenu)
        if sel:
            ci = (0, self.hwnd_parent, sel - id_cmd_first, None, None, 0, 0, 0)
            cm.InvokeCommand(ci)

    def OnSize(self, hwnd, msg, wparam, lparam):
        if False:
            while True:
                i = 10
        if self.hwnd_child is not None:
            x = win32api.LOWORD(lparam)
            y = win32api.HIWORD(lparam)
            win32gui.MoveWindow(self.hwnd_child, 0, 0, x, y, False)

class ScintillaShellView:
    _public_methods_ = shellcon.IShellView_Methods
    _com_interfaces_ = [pythoncom.IID_IOleWindow, shell.IID_IShellView]

    def __init__(self, hwnd, filename, lineno=None):
        if False:
            print('Hello World!')
        self.filename = filename
        self.lineno = lineno
        self.hwnd_parent = hwnd
        self.hwnd = None

    def _SendSci(self, msg, wparam=0, lparam=0):
        if False:
            print('Hello World!')
        return win32gui.SendMessage(self.hwnd, msg, wparam, lparam)

    def CreateViewWindow(self, prev, settings, browser, rect):
        if False:
            return 10
        print('ScintillaShellView.CreateViewWindow', prev, settings, browser, rect)
        try:
            win32api.GetModuleHandle('Scintilla.dll')
        except win32api.error:
            for p in sys.path:
                fname = os.path.join(p, 'Scintilla.dll')
                if not os.path.isfile(fname):
                    fname = os.path.join(p, 'Build', 'Scintilla.dll')
                if os.path.isfile(fname):
                    win32api.LoadLibrary(fname)
                    break
            else:
                raise RuntimeError("Can't find scintilla!")
        style = win32con.WS_CHILD | win32con.WS_VSCROLL | win32con.WS_HSCROLL | win32con.WS_CLIPCHILDREN | win32con.WS_VISIBLE
        self.hwnd = win32gui.CreateWindow('Scintilla', 'Scintilla', style, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], self.hwnd_parent, 1000, 0, None)
        message_map = {win32con.WM_SIZE: self.OnSize}
        file_data = file(self.filename, 'U').read()
        self._SetupLexer()
        self._SendSci(scintillacon.SCI_ADDTEXT, len(file_data), file_data)
        if self.lineno is not None:
            self._SendSci(scintillacon.SCI_GOTOLINE, self.lineno)
        print("Scintilla's hwnd is", self.hwnd)

    def _SetupLexer(self):
        if False:
            return 10
        h = self.hwnd
        styles = [((0, 0, 200, 0, 8421504), None, scintillacon.SCE_P_DEFAULT), ((0, 2, 200, 0, 32768), None, scintillacon.SCE_P_COMMENTLINE), ((0, 2, 200, 0, 8421504), None, scintillacon.SCE_P_COMMENTBLOCK), ((0, 0, 200, 0, 8421376), None, scintillacon.SCE_P_NUMBER), ((0, 0, 200, 0, 32896), None, scintillacon.SCE_P_STRING), ((0, 0, 200, 0, 32896), None, scintillacon.SCE_P_CHARACTER), ((0, 0, 200, 0, 32896), None, scintillacon.SCE_P_TRIPLE), ((0, 0, 200, 0, 32896), None, scintillacon.SCE_P_TRIPLEDOUBLE), ((0, 0, 200, 0, 0), 32896, scintillacon.SCE_P_STRINGEOL), ((0, 1, 200, 0, 8388608), None, scintillacon.SCE_P_WORD), ((0, 1, 200, 0, 16711680), None, scintillacon.SCE_P_CLASSNAME), ((0, 1, 200, 0, 8421376), None, scintillacon.SCE_P_DEFNAME), ((0, 0, 200, 0, 0), None, scintillacon.SCE_P_OPERATOR), ((0, 0, 200, 0, 0), None, scintillacon.SCE_P_IDENTIFIER)]
        self._SendSci(scintillacon.SCI_SETLEXER, scintillacon.SCLEX_PYTHON, 0)
        self._SendSci(scintillacon.SCI_SETSTYLEBITS, 5)
        baseFormat = (-402653169, 0, 200, 0, 0, 0, 49, 'Courier New')
        for (f, bg, stylenum) in styles:
            self._SendSci(scintillacon.SCI_STYLESETFORE, stylenum, f[4])
            self._SendSci(scintillacon.SCI_STYLESETFONT, stylenum, baseFormat[7])
            if f[1] & 1:
                self._SendSci(scintillacon.SCI_STYLESETBOLD, stylenum, 1)
            else:
                self._SendSci(scintillacon.SCI_STYLESETBOLD, stylenum, 0)
            if f[1] & 2:
                self._SendSci(scintillacon.SCI_STYLESETITALIC, stylenum, 1)
            else:
                self._SendSci(scintillacon.SCI_STYLESETITALIC, stylenum, 0)
            self._SendSci(scintillacon.SCI_STYLESETSIZE, stylenum, int(baseFormat[2] / 20))
            if bg is not None:
                self._SendSci(scintillacon.SCI_STYLESETBACK, stylenum, bg)
            self._SendSci(scintillacon.SCI_STYLESETEOLFILLED, stylenum, 1)

    def GetWindow(self):
        if False:
            for i in range(10):
                print('nop')
        return self.hwnd

    def UIActivate(self, activate_state):
        if False:
            while True:
                i = 10
        print('OnActivate')

    def DestroyViewWindow(self):
        if False:
            return 10
        win32gui.DestroyWindow(self.hwnd)
        self.hwnd = None
        print('Destroyed scintilla window')

    def TranslateAccelerator(self, msg):
        if False:
            while True:
                i = 10
        return winerror.S_FALSE

    def OnSize(self, hwnd, msg, wparam, lparam):
        if False:
            while True:
                i = 10
        x = win32api.LOWORD(lparam)
        y = win32api.HIWORD(lparam)
        win32gui.MoveWindow(self.hwnd, 0, 0, x, y, False)

def DllRegisterServer():
    if False:
        return 10
    import winreg
    key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\' + ShellFolderRoot._reg_clsid_)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellFolderRoot._reg_desc_)
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 'CLSID\\' + ShellFolderRoot._reg_clsid_ + '\\ShellFolder')
    attr = shellcon.SFGAO_FOLDER | shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_BROWSABLE
    import struct
    s = struct.pack('i', attr)
    winreg.SetValueEx(key, 'Attributes', 0, winreg.REG_BINARY, s)
    print(ShellFolderRoot._reg_desc_, 'registration complete.')

def DllUnregisterServer():
    if False:
        while True:
            i = 10
    import winreg
    try:
        key = winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\' + ShellFolderRoot._reg_clsid_)
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    print(ShellFolderRoot._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(ShellFolderRoot, debug=debug, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)