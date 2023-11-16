import win32con
import win32ui
from pywin.framework import interact
from pywin.mfc import dialog
buttonControlMap = {win32ui.IDC_BUTTON1: win32ui.IDC_EDIT1, win32ui.IDC_BUTTON2: win32ui.IDC_EDIT2, win32ui.IDC_BUTTON3: win32ui.IDC_EDIT3}

class OptionsPropPage(dialog.PropertyPage):

    def __init__(self):
        if False:
            print('Hello World!')
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_IDE)
        self.AddDDX(win32ui.IDC_CHECK1, 'bShowAtStartup')
        self.AddDDX(win32ui.IDC_CHECK2, 'bDocking')
        self.AddDDX(win32ui.IDC_EDIT4, 'MRUSize', 'i')

    def OnInitDialog(self):
        if False:
            return 10
        edit = self.GetDlgItem(win32ui.IDC_EDIT1)
        format = eval(win32ui.GetProfileVal(interact.sectionProfile, interact.STYLE_INTERACTIVE_PROMPT, str(interact.formatInput)))
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText('Input Text')
        edit = self.GetDlgItem(win32ui.IDC_EDIT2)
        format = eval(win32ui.GetProfileVal(interact.sectionProfile, interact.STYLE_INTERACTIVE_OUTPUT, str(interact.formatOutput)))
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText('Output Text')
        edit = self.GetDlgItem(win32ui.IDC_EDIT3)
        format = eval(win32ui.GetProfileVal(interact.sectionProfile, interact.STYLE_INTERACTIVE_ERROR, str(interact.formatOutputError)))
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText('Error Text')
        self['bShowAtStartup'] = interact.LoadPreference('Show at startup', 1)
        self['bDocking'] = interact.LoadPreference('Docking', 0)
        self['MRUSize'] = win32ui.GetProfileVal('Settings', 'Recent File List Size', 10)
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON1)
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON2)
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON3)
        spinner = self.GetDlgItem(win32ui.IDC_SPIN1)
        spinner.SetRange(1, 16)
        return dialog.PropertyPage.OnInitDialog(self)

    def HandleCharFormatChange(self, id, code):
        if False:
            return 10
        if code == win32con.BN_CLICKED:
            editId = buttonControlMap.get(id)
            assert editId is not None, 'Format button has no associated edit control'
            editControl = self.GetDlgItem(editId)
            existingFormat = editControl.GetDefaultCharFormat()
            flags = win32con.CF_SCREENFONTS
            d = win32ui.CreateFontDialog(existingFormat, flags, None, self)
            if d.DoModal() == win32con.IDOK:
                cf = d.GetCharFormat()
                editControl.SetDefaultCharFormat(cf)
                self.SetModified(1)
            return 0

    def OnOK(self):
        if False:
            for i in range(10):
                print('nop')
        controlAttrs = [(win32ui.IDC_EDIT1, interact.STYLE_INTERACTIVE_PROMPT), (win32ui.IDC_EDIT2, interact.STYLE_INTERACTIVE_OUTPUT), (win32ui.IDC_EDIT3, interact.STYLE_INTERACTIVE_ERROR)]
        for (id, key) in controlAttrs:
            control = self.GetDlgItem(id)
            fmt = control.GetDefaultCharFormat()
            win32ui.WriteProfileVal(interact.sectionProfile, key, str(fmt))
        interact.SavePreference('Show at startup', self['bShowAtStartup'])
        interact.SavePreference('Docking', self['bDocking'])
        win32ui.WriteProfileVal('Settings', 'Recent File List Size', self['MRUSize'])
        return 1

    def ChangeFormat(self, fmtAttribute, fmt):
        if False:
            for i in range(10):
                print('nop')
        dlg = win32ui.CreateFontDialog(fmt)
        if dlg.DoModal() != win32con.IDOK:
            return None
        return dlg.GetCharFormat()

    def OnFormatTitle(self, command, code):
        if False:
            return 10
        fmt = self.GetFormat(interact.formatTitle)
        if fmt:
            formatTitle = fmt
            SaveFontPreferences()

    def OnFormatInput(self, command, code):
        if False:
            i = 10
            return i + 15
        global formatInput
        fmt = self.GetFormat(formatInput)
        if fmt:
            formatInput = fmt
            SaveFontPreferences()

    def OnFormatOutput(self, command, code):
        if False:
            print('Hello World!')
        global formatOutput
        fmt = self.GetFormat(formatOutput)
        if fmt:
            formatOutput = fmt
            SaveFontPreferences()

    def OnFormatError(self, command, code):
        if False:
            while True:
                i = 10
        global formatOutputError
        fmt = self.GetFormat(formatOutputError)
        if fmt:
            formatOutputError = fmt
            SaveFontPreferences()