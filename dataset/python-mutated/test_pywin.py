import argparse
import os
import re
import sys
import time
import traceback
import types
import unittest
import weakref
from unittest import mock
import __main__
import pywin
import win32api
import win32con as wc
import win32gui
import win32ui
from pywin.framework import scriptutils
_clock = time.perf_counter
user_interaction = getattr(__main__, 'user_interaction', False)
file_abs = os.path.abspath(__file__)
src_dir = os.path.dirname(file_abs)
pywin_path = pywin.__path__[0]
pythonwinpy_path = os.path.dirname(pywin_path) + '\\start_pythonwin.py'
Object = argparse.Namespace
_indebugger = 'pywin.debugger' in sys.modules

def read_file(*args, **kw):
    if False:
        i = 10
        return i + 15
    with open(*args, **kw) as f:
        return f.read()

class T(unittest.TestCase):
    """Runs and tests the Pythonwin app & win32ui directly inside this process
    without booting a Pythonwin.exe process."""

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        from pywin.framework.intpyapp import thisApp
        cls.app = thisApp
        cls.std_oe_orig = (sys.stdout, sys.stderr)

        def _restore_oe():
            if False:
                for i in range(10):
                    print('nop')
            (sys.stdout, sys.stderr) = cls.std_oe_orig
        if sys.version_info >= (3, 8):
            cls.addClassCleanup(_restore_oe)
        sys.argv[1:] = ['/new', src_dir + '\\_dbgscript.py']
        if not _indebugger:
            thisApp.InitInstance()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        global teared_down
        teared_down = 1
        if user_interaction:
            print('-- Interact, then close the window for continuing the tests!')
            cls.app.Run()
        if not _indebugger:
            cls.app.frame.DestroyWindow()
            win32api.PostQuitMessage()
            win32gui.PumpWaitingMessages()
            cls.app.ExitInstance()
        (sys.stdout, sys.stderr) = cls.std_oe_orig

    def test_1_pydocs_and_finddlg(self):
        if False:
            return 10
        mf = win32ui.GetMainFrame()
        some_fn = src_dir + '\\_dbgscript.py'
        assert some_fn != file_abs
        scriptutils.JumpToDocument(some_fn)
        a = scriptutils.GetActiveFileName()
        assert some_fn == a
        v = scriptutils.GetActiveEditControl()
        s = read_file(some_fn, encoding='latin-1', newline='\r\n')
        assert s == v.GetTextRange(), 'doc encoding not detected'
        scriptutils.JumpToDocument(__file__)
        if user_interaction:
            win32ui.MessageBox(f'Hello from test_pydocs() args={sys.argv} {os.getcwd()}')
        v = scriptutils.GetActiveEditControl()
        assert file_abs == v.GetDocument().GetPathName()
        t = v.GetTextRange()
        testpat = 'self.app = thisApp'
        assert testpat in t
        assert read_file(__file__, encoding='utf-8', newline='\r\n') == t
        v.SetSel(0)
        assert v.GetSel() == (0, 0)
        import pywin.scintilla.find
        win32ui.PumpWaitingMessages(0, -1)
        win32gui.PumpWaitingMessages()
        m = mf.GetMenu()
        ix = 1
        es = m.GetMenuString(ix, wc.MF_BYPOSITION)
        if '&Edit' != es:
            ix += 1
            es = m.GetMenuString(ix, wc.MF_BYPOSITION)
        assert '&Edit' == es
        editm = m.GetSubMenu(ix)
        assert editm.GetMenuItemCount() > 10
        for i in range(14):
            s = editm.GetMenuString(i, wc.MF_BYPOSITION)
            if s.startswith('R&eplace'):
                break
        else:
            assert 0, 'Replace menu entry not found'
        replace_id = editm.GetMenuItemID(i)
        win32gui.PumpWaitingMessages()
        v.SendMessage(wc.WM_COMMAND, replace_id)
        d = pywin.scintilla.find.curDialog
        d.editFindText.SetWindowText(testpat)
        d.OnFindNext(0, 0)
        (s, e) = v.GetSel()
        assert e - s == len(testpat) and s > 0

    def test_browseobj(self):
        if False:
            for i in range(10):
                print('nop')
        'Test object browser'
        import pywin.tools.browser
        o = Object()
        Browse = pywin.tools.browser.Browse

        def t_Browse(*args):
            if False:
                i = 10
                return i + 15
            o.dlg = Browse(*args)
            return o.dlg
        if __name__ != '__main__':
            setattr(__main__, __class__.__qualname__, __class__)
        with mock.patch('pywin.mfc.dialog.GetSimpleInput', lambda *args: __class__.__qualname__), mock.patch('pywin.tools.browser.Browse', t_Browse):
            self.app.OnViewBrowse(0, 0)
        hl = o.dlg.hier_list
        assert len(hl.itemHandleMap) > 10
        assert hl.listControl.GetCount() > 10
        item = hl.GetSelectedItem()
        assert 'TestCase' in str(hl.listControl.GetItem(item))
        assert 'TestCase' in hl.ItemFromHandle(item).GetText()
        item2 = hl.listControl.GetNextVisibleItem(item)
        assert 'Runs and tests' in str(hl.listControl.GetItem(item2))

    def test_options_propsheet(self):
        if False:
            while True:
                i = 10
        'Check Pythonwin options property sheet'
        lres = []

        def t_DoModal(self):
            if False:
                return 10
            self.CreateWindow()
            p = self.GetPage(4)
            self.SetActivePage(p)
            p = self.GetPage(4)
            assert p._DoButDefaultFont
            assert p.GetDlgItem(win32ui.IDC_RADIO1).GetCheck() ^ p.GetDlgItem(win32ui.IDC_RADIO2).GetCheck()
            assert p.listbox.GetCount() >= 16
            assert p.GetSelectedStyle().name
            lres.append('done')
            w_obj = weakref.ref(p._obj_)
            assert w_obj()
            self.DestroyWindow()
            assert p._obj_ is None
            assert self._obj_ is None
        with mock.patch('pywin.mfc.dialog.PropertySheet.DoModal', t_DoModal):
            self.app.OnViewOptions(0, 0)
            assert lres

    def test_ctrls(self):
        if False:
            for i in range(10):
                print('nop')
        from pywin.mfc import dialog
        _ds = wc.WS_MINIMIZEBOX | wc.WS_DLGFRAME | wc.DS_MODALFRAME | wc.WS_POPUP | wc.WS_VISIBLE | wc.WS_CAPTION | wc.WS_SYSMENU | wc.DS_SETFONT
        _bs = wc.BS_PUSHBUTTON | wc.WS_TABSTOP | wc.WS_CHILD | wc.WS_VISIBLE
        DT = [['Test Dialog', (0, 0, 100, 100), _ds, None, (8, 'MS SansSerif')], [128, 'Close', wc.IDCANCEL, (5, 80, 50, 13), _bs]]
        mf = win32ui.GetMainFrame()
        d = dialog.Dialog(DT)
        d.CreateWindow(mf)
        self.addCleanup(lambda : d._obj_ and d.DestroyWindow())
        slider = win32ui.CreateSliderCtrl()
        _cst = wc.WS_TABSTOP | wc.WS_VISIBLE | wc.WS_CHILD
        slider.CreateWindow(_cst, (0, 10, 200, 40), d, 100)
        win32gui.PumpWaitingMessages()
        (mi, ma) = slider.GetRange()
        assert slider.GetPos() == 0
        slider.SetPos(20)
        assert slider.GetPos() == 20
        pc = win32ui.CreateProgressCtrl()
        pc.CreateWindow(_cst, (0, 35, 200, 55), d, 100)
        pc.SetRange(0, 50)
        pc.SetPos(55)
        pc.StepIt() == 55
        edit = win32ui.CreateEdit()
        edit.CreateWindow(_cst | wc.WS_BORDER, (5, 60, 100, 80), d, 101)
        assert d.GetDlgItem(101) is edit
        d.DestroyWindow()
        assert d._obj_ is None

    def test_dc(self):
        if False:
            return 10
        from pywin.mfc import window
        o = Object(cnt_onpaint=0, cnt_onsize=0, cnt_ondestroy=0)
        font = win32ui.CreateFont({'name': 'Arial', 'height': 32})
        pen = win32ui.CreatePen(wc.PS_SOLID, 5, 1179426)
        brush = win32ui.GetHalftoneBrush()
        brush2 = win32ui.CreateBrush()
        brush2.CreateSolidBrush(win32api.GetSysColor(wc.COLOR_HOTLIGHT))

        class PaintWnd(window.MDIChildWnd):

            def Create(self, title, rect=None, parent=None):
                if False:
                    while True:
                        i = 10
                style = wc.WS_CHILD | wc.WS_VISIBLE | wc.WS_OVERLAPPEDWINDOW
                self.CreateWindow(None, title, style, rect, parent)
                self.HookMessage(self.OnDestroy, wc.WM_DESTROY)
                self.HookMessage(self.OnSize, wc.WM_SIZE)

            def OnSize(self, msg):
                if False:
                    return 10
                o.cnt_onsize += 1

            def OnDestroy(self, msg):
                if False:
                    for i in range(10):
                        print('nop')
                o.cnt_ondestroy += 1

            def OnPaint(self):
                if False:
                    i = 10
                    return i + 15
                try:
                    (dc, paintStruct) = self.BeginPaint()
                    dc.FillSolidRect(self.GetClientRect(), win32api.RGB(255, 0, 0))
                    r = self.GetClientRect()
                    dc.Pie(r[0], r[1], r[2], r[3], 0, 0, r[2], r[3] // 2)
                    dc.SelectObject(font)
                    dc.SelectObject(pen)
                    dc.SelectObject(brush2)
                    dc.SetPolyFillMode(wc.WINDING)
                    dc.Rectangle((110, 100, 180, 160))
                    dc.SelectObject(brush)
                    dc.Polygon([(20, 20), (80, 30), (90, 200), (20, 20)])
                    dc.MoveTo(140, 10)
                    dc.LineTo(180, 30)
                    dc.SetTextColor(16711714)
                    dc.SetBkColor(658188)
                    dc.SetBkMode(wc.TRANSPARENT)
                    dc.SetTextAlign(wc.TA_LEFT | wc.TA_BASELINE)
                    dc.TextOut(60, 100, 'TextOut by test_dc()')
                    dc.DrawText('DrawText', (10, 30, 190, 160), wc.DT_NOCLIP)
                    dc.SetMapMode(wc.MM_ANISOTROPIC)
                    dc.SetViewportOrg((40, 40))
                    dc.Polyline([(20, 20), (80, 30), (90, 200), (20, 20)])
                    self.EndPaint(paintStruct)
                    o.cnt_onpaint += 1
                except Exception as e:
                    o.exc = e
                    del __class__.OnPaint
                    raise
        w = PaintWnd()
        w.Create('Test Paint MDI Child')
        self.addCleanup(lambda : o.cnt_ondestroy or w.DestroyWindow())
        win32gui.PumpWaitingMessages()
        dc = w.GetDC()
        assert o.cnt_onpaint > 0, ''.join(traceback.format_exception(None, o.exc, o.exc.__traceback__))
        pix = dc.GetPixel(1, 1)
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(dc, 30, 30)
        dcb = dc.CreateCompatibleDC(dc)
        dcb.SelectObject(bmp)
        dcb.BitBlt((0, 0), (30, 30), dc, (0, 0), wc.SRCCOPY)
        sbits = bmp.GetBitmapBits(0)
        assert any(sbits[:4])
        w.ReleaseDC(dc)
        assert pix == 255
        assert o.cnt_ondestroy == 0
        w.DestroyWindow()
        assert o.cnt_ondestroy == 1

    def test_ia(self):
        if False:
            return 10
        'Test interactive, run, autocomplete, exec'
        ia = pywin.framework.interact.edit.currentView
        fn = src_dir + '\\_dbgscript.py'
        mf = win32ui.GetMainFrame()
        scriptutils.JumpToDocument(fn)
        cmGo = win32ui.IDC_DBG_GO
        mf.SendMessage(wc.WM_COMMAND, cmGo)
        assert __main__.aa == 33 == ia.interp.globals['aa']
        assert __main__.ff() == 132
        ia.SetFocus()
        ia.EnsureNoPrompt()
        ia.AppendToPrompt(['if 1:'])
        ia.ProcessEnterEvent(None)
        ia.ReplaceSel('CC')
        tail1 = ia.GetTextRange(ia.GetTextLength() - 20)
        assert tail1.endswith('... \tCC'), 'wrong auto-indent: %r' % tail1
        ia.SendMessage(wc.WM_KEYDOWN, win32api.VkKeyScan('.'))
        ia.SendMessage(wc.WM_KEYUP, win32api.VkKeyScan('.'))
        ia.SendMessage(wc.WM_KEYDOWN, wc.VK_TAB)
        ia.SendMessage(wc.WM_KEYUP, wc.VK_TAB)
        tail2 = ia.GetTextRange(ia.GetTextLength() - 20)
        assert tail2.endswith('... \tCC.cc'), 'wrong auto-complete: %r' % tail2
        ia.ProcessEnterEvent(None)
        ia.SendMessage(wc.WM_KEYDOWN, wc.VK_RETURN)
        ia.SendMessage(wc.WM_KEYUP, wc.VK_RETURN)
        execd = ia.GetTextRange(ia.GetTextLength() - 20)
        assert '\n44' in execd, 'wrong result: %r' % execd
        ia.SetFocus()
        ia.SCICallTipCancel()
        ia.AppendToPrompt(['ff'])
        ss_vk = win32api.VkKeyScan('(')
        shift = ss_vk & 256
        t_GKS = lambda key: (key == wc.VK_SHIFT and shift) and 32768 or 0
        with mock.patch('win32api.GetKeyState', t_GKS):
            assert not ia.SCICallTipActive()
            ia.SendMessage(wc.WM_KEYDOWN, ss_vk & 255)
            ia.SendMessage(wc.WM_CHAR, ord('('))
            ia.SendMessage(wc.WM_KEYUP, ss_vk & 255)
        assert ia.SCICallTipActive()
        if ia.GetSel()[1] == ia.GetTextLength():
            ia.SendMessage(wc.WM_CHAR, ord(')'))
        ia.GotoEndOfFileEvent(None)
        ia.SendMessage(wc.WM_KEYDOWN, wc.VK_RETURN)
        ia.SendMessage(wc.WM_KEYUP, wc.VK_RETURN)
        execd = ia.GetTextRange(ia.GetTextLength() - 20)
        assert '\n132' in execd, execd

    def test_docedit(self):
        if False:
            for i in range(10):
                print('nop')
        import tempfile
        import pywin.scintilla.IDLEenvironment

        def t_print(*args):
            if False:
                return 10
            assert 'ERROR' not in str(args)
            assert 0, 'should not print at all'
        with mock.patch('builtins.print', t_print):
            pywin.scintilla.IDLEenvironment.test()
        ed = scriptutils.GetActiveEditControl()
        doc = ed.GetDocument()
        assert 'hi there' in ed.GetTextRange()
        assert doc.IsModified()
        ed.SetWindowText('')
        doc.SetModifiedFlag(0)
        ed.SCIAddText('if 1:')
        ed.EnterKeyEvent(None)
        ed.SCIAddText('CC')
        ed.SendMessage(wc.WM_KEYDOWN, wc.VK_RETURN)
        ed.SendMessage(wc.WM_KEYUP, wc.VK_RETURN)
        s = ed.GetTextRange()
        assert re.match('(?m)if 1:\\r\\n[ \\t]+CC\\r\\n[ \\t]+\\r\\n$', s), 'no auto-indent'
        (fh, tfn) = tempfile.mkstemp(suffix='.py', prefix='pywintest-')
        os.close(fh)
        self.addCleanup(lambda : os.remove(tfn))
        doc.OnSaveDocument(tfn)
        r = read_file(tfn, 'rb').decode()
        assert s == r
        doc.OnCloseDocument()

    def test_debugger(self):
        if False:
            i = 10
            return i + 15
        import pywin.debugger
        import pywin.framework.dbgcommands
        fn = src_dir + '\\_dbgscript.py'
        deb = pywin.debugger.GetDebugger()
        mf = win32ui.GetMainFrame()
        scriptutils.JumpToDocument(fn)
        v = scriptutils.GetActiveEditControl()
        deb.clear_all_breaks()
        win32gui.PumpWaitingMessages()
        src = v.GetTextRange()
        assert 'aa = 33' in src

        def getlno(s):
            if False:
                return 10
            return src[:src.index(s)].count('\n') + 1
        cmGo = win32ui.IDC_DBG_GO
        cmClose = win32ui.IDC_DBG_CLOSE
        deb.set_break(fn, getlno('aa = 22'))
        deb.set_break(fn, getlno('aa = 77'))
        cmds_brk_next = [cmGo, cmClose]
        self.addCleanup(lambda : (deb.clear_all_breaks(), deb.UpdateAllLineStates()))
        obj = Object(brk_linenos=[])
        GUIAboutToBreak = deb.GUIAboutToBreak

        def t_brk(self):
            if False:
                return 10
            obj.brk_linenos.append(deb.curframe.f_lineno)
            mf.PostMessage(wc.WM_COMMAND, cmds_brk_next.pop(0))
            GUIAboutToBreak()
        dmod = types.ModuleType('__main__', 'debugger test main')
        with mock.patch('pywin.framework.scriptutils.__main__', dmod), mock.patch('pywin.debugger.debugger.Debugger.GUIAboutToBreak', t_brk):
            mf.SendMessage(wc.WM_COMMAND, cmGo)
        assert not cmds_brk_next, 'break commands remaining'
        assert obj.brk_linenos[0] == getlno('aa = 22')
        assert obj.brk_linenos[1] == getlno('aa = 77')
        assert dmod.aa == 22
if __name__ == '__main__':
    if _indebugger:
        t = T('test_docedit')
        ts = unittest.TestSuite((t,))
        _tests = ts._tests[:]
        r = ts.debug()
        assert teared_down
        print(_tests, 'ok!')
        sys.exit()
    p = argparse.ArgumentParser(description='Tests for Pythonwin / win32ui', add_help=False)
    p.add_argument('-user-interaction', '-i', action='store_true', help='Include tests which require user interaction')
    (args, remains) = p.parse_known_args()
    user_interaction = args.user_interaction
    if '-h' in sys.argv or '--help' in sys.argv:
        p.print_help()
    unittest.main(argv=sys.argv[:1] + remains)