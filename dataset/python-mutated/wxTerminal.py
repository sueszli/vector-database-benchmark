import codecs
from serial.tools.miniterm import unichr
import serial
import threading
import wx
import wx.lib.newevent
import wxSerialConfigDialog
try:
    unichr
except NameError:
    unichr = chr
(SerialRxEvent, EVT_SERIALRX) = wx.lib.newevent.NewEvent()
SERIALRX = wx.NewEventType()
ID_CLEAR = wx.NewId()
ID_SAVEAS = wx.NewId()
ID_SETTINGS = wx.NewId()
ID_TERM = wx.NewId()
ID_EXIT = wx.NewId()
ID_RTS = wx.NewId()
ID_DTR = wx.NewId()
NEWLINE_CR = 0
NEWLINE_LF = 1
NEWLINE_CRLF = 2

class TerminalSetup:
    """
    Placeholder for various terminal settings. Used to pass the
    options to the TerminalSettingsDialog.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.echo = False
        self.unprintable = False
        self.newline = NEWLINE_CRLF

class TerminalSettingsDialog(wx.Dialog):
    """Simple dialog with common terminal settings like echo, newline mode."""

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        self.settings = kwds['settings']
        del kwds['settings']
        kwds['style'] = wx.DEFAULT_DIALOG_STYLE
        wx.Dialog.__init__(self, *args, **kwds)
        self.checkbox_echo = wx.CheckBox(self, -1, 'Local Echo')
        self.checkbox_unprintable = wx.CheckBox(self, -1, 'Show unprintable characters')
        self.radio_box_newline = wx.RadioBox(self, -1, 'Newline Handling', choices=['CR only', 'LF only', 'CR+LF'], majorDimension=0, style=wx.RA_SPECIFY_ROWS)
        self.sizer_4_staticbox = wx.StaticBox(self, -1, 'Input/Output')
        self.button_ok = wx.Button(self, wx.ID_OK, '')
        self.button_cancel = wx.Button(self, wx.ID_CANCEL, '')
        self.__set_properties()
        self.__do_layout()
        self.__attach_events()
        self.checkbox_echo.SetValue(self.settings.echo)
        self.checkbox_unprintable.SetValue(self.settings.unprintable)
        self.radio_box_newline.SetSelection(self.settings.newline)

    def __set_properties(self):
        if False:
            print('Hello World!')
        self.SetTitle('Terminal Settings')
        self.radio_box_newline.SetSelection(0)
        self.button_ok.SetDefault()

    def __do_layout(self):
        if False:
            return 10
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_4_staticbox.Lower()
        sizer_4 = wx.StaticBoxSizer(self.sizer_4_staticbox, wx.VERTICAL)
        sizer_4.Add(self.checkbox_echo, 0, wx.ALL, 4)
        sizer_4.Add(self.checkbox_unprintable, 0, wx.ALL, 4)
        sizer_4.Add(self.radio_box_newline, 0, 0, 0)
        sizer_2.Add(sizer_4, 0, wx.EXPAND, 0)
        sizer_3.Add(self.button_ok, 0, 0, 0)
        sizer_3.Add(self.button_cancel, 0, 0, 0)
        sizer_2.Add(sizer_3, 0, wx.ALL | wx.ALIGN_RIGHT, 4)
        self.SetSizer(sizer_2)
        sizer_2.Fit(self)
        self.Layout()

    def __attach_events(self):
        if False:
            i = 10
            return i + 15
        self.Bind(wx.EVT_BUTTON, self.OnOK, id=self.button_ok.GetId())
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=self.button_cancel.GetId())

    def OnOK(self, events):
        if False:
            print('Hello World!')
        'Update data with new values and close dialog.'
        self.settings.echo = self.checkbox_echo.GetValue()
        self.settings.unprintable = self.checkbox_unprintable.GetValue()
        self.settings.newline = self.radio_box_newline.GetSelection()
        self.EndModal(wx.ID_OK)

    def OnCancel(self, events):
        if False:
            return 10
        'Do not update data but close dialog.'
        self.EndModal(wx.ID_CANCEL)

class TerminalFrame(wx.Frame):
    """Simple terminal program for wxPython"""

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        self.serial = serial.Serial()
        self.serial.timeout = 0.5
        self.settings = TerminalSetup()
        self.thread = None
        self.alive = threading.Event()
        kwds['style'] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.frame_terminal_menubar = wx.MenuBar()
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(ID_CLEAR, '&Clear', '', wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(ID_SAVEAS, '&Save Text As...', '', wx.ITEM_NORMAL)
        wxglade_tmp_menu.AppendSeparator()
        wxglade_tmp_menu.Append(ID_TERM, '&Terminal Settings...', '', wx.ITEM_NORMAL)
        wxglade_tmp_menu.AppendSeparator()
        wxglade_tmp_menu.Append(ID_EXIT, '&Exit', '', wx.ITEM_NORMAL)
        self.frame_terminal_menubar.Append(wxglade_tmp_menu, '&File')
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(ID_RTS, 'RTS', '', wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(ID_DTR, '&DTR', '', wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(ID_SETTINGS, '&Port Settings...', '', wx.ITEM_NORMAL)
        self.frame_terminal_menubar.Append(wxglade_tmp_menu, 'Serial Port')
        self.SetMenuBar(self.frame_terminal_menubar)
        self.text_ctrl_output = wx.TextCtrl(self, -1, '', style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.__set_properties()
        self.__do_layout()
        self.Bind(wx.EVT_MENU, self.OnClear, id=ID_CLEAR)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=ID_SAVEAS)
        self.Bind(wx.EVT_MENU, self.OnTermSettings, id=ID_TERM)
        self.Bind(wx.EVT_MENU, self.OnExit, id=ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnRTS, id=ID_RTS)
        self.Bind(wx.EVT_MENU, self.OnDTR, id=ID_DTR)
        self.Bind(wx.EVT_MENU, self.OnPortSettings, id=ID_SETTINGS)
        self.__attach_events()
        self.OnPortSettings(None)
        if not self.alive.is_set():
            self.Close()

    def StartThread(self):
        if False:
            print('Hello World!')
        'Start the receiver thread'
        self.thread = threading.Thread(target=self.ComPortThread)
        self.thread.daemon = True
        self.alive.set()
        self.thread.start()
        self.serial.rts = True
        self.serial.dtr = True
        self.frame_terminal_menubar.Check(ID_RTS, self.serial.rts)
        self.frame_terminal_menubar.Check(ID_DTR, self.serial.dtr)

    def StopThread(self):
        if False:
            print('Hello World!')
        "Stop the receiver thread, wait until it's finished."
        if self.thread is not None:
            self.alive.clear()
            self.thread.join()
            self.thread = None

    def __set_properties(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetTitle('Serial Terminal')
        self.SetSize((546, 383))
        self.text_ctrl_output.SetFont(wx.Font(9, wx.MODERN, wx.NORMAL, wx.NORMAL, 0, ''))

    def __do_layout(self):
        if False:
            return 10
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.text_ctrl_output, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        self.Layout()

    def __attach_events(self):
        if False:
            i = 10
            return i + 15
        self.Bind(wx.EVT_MENU, self.OnClear, id=ID_CLEAR)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=ID_SAVEAS)
        self.Bind(wx.EVT_MENU, self.OnExit, id=ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnPortSettings, id=ID_SETTINGS)
        self.Bind(wx.EVT_MENU, self.OnTermSettings, id=ID_TERM)
        self.text_ctrl_output.Bind(wx.EVT_CHAR, self.OnKey)
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKey)
        self.Bind(EVT_SERIALRX, self.OnSerialRead)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnExit(self, event):
        if False:
            i = 10
            return i + 15
        'Menu point Exit'
        self.Close()

    def OnClose(self, event):
        if False:
            return 10
        'Called on application shutdown.'
        self.StopThread()
        self.serial.close()
        self.Destroy()

    def OnSaveAs(self, event):
        if False:
            print('Hello World!')
        'Save contents of output window.'
        with wx.FileDialog(None, 'Save Text As...', '.', '', 'Text File|*.txt|All Files|*', wx.SAVE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                filename = dlg.GetPath()
                with codecs.open(filename, 'w', encoding='utf-8') as f:
                    text = self.text_ctrl_output.GetValue().encode('utf-8')
                    f.write(text)

    def OnClear(self, event):
        if False:
            i = 10
            return i + 15
        'Clear contents of output window.'
        self.text_ctrl_output.Clear()

    def OnPortSettings(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show the port settings dialog. The reader thread is stopped for the\n        settings change.\n        '
        if event is not None:
            self.StopThread()
            self.serial.close()
        ok = False
        while not ok:
            with wxSerialConfigDialog.SerialConfigDialog(self, -1, '', show=wxSerialConfigDialog.SHOW_BAUDRATE | wxSerialConfigDialog.SHOW_FORMAT | wxSerialConfigDialog.SHOW_FLOW, serial=self.serial) as dialog_serial_cfg:
                dialog_serial_cfg.CenterOnParent()
                result = dialog_serial_cfg.ShowModal()
            if result == wx.ID_OK or event is not None:
                try:
                    self.serial.open()
                except serial.SerialException as e:
                    with wx.MessageDialog(self, str(e), 'Serial Port Error', wx.OK | wx.ICON_ERROR) as dlg:
                        dlg.ShowModal()
                else:
                    self.StartThread()
                    self.SetTitle('Serial Terminal on {} [{},{},{},{}{}{}]'.format(self.serial.portstr, self.serial.baudrate, self.serial.bytesize, self.serial.parity, self.serial.stopbits, ' RTS/CTS' if self.serial.rtscts else '', ' Xon/Xoff' if self.serial.xonxoff else ''))
                    ok = True
            else:
                self.alive.clear()
                ok = True

    def OnTermSettings(self, event):
        if False:
            print('Hello World!')
        '        Menu point Terminal Settings. Show the settings dialog\n        with the current terminal settings.\n        '
        with TerminalSettingsDialog(self, -1, '', settings=self.settings) as dialog:
            dialog.CenterOnParent()
            dialog.ShowModal()

    def OnKey(self, event):
        if False:
            while True:
                i = 10
        '        Key event handler. If the key is in the ASCII range, write it to the\n        serial port. Newline handling and local echo is also done here.\n        '
        code = event.GetUnicodeKey()
        if code == 13:
            if self.settings.echo:
                self.text_ctrl_output.AppendText('\n')
            if self.settings.newline == NEWLINE_CR:
                self.serial.write(b'\r')
            elif self.settings.newline == NEWLINE_LF:
                self.serial.write(b'\n')
            elif self.settings.newline == NEWLINE_CRLF:
                self.serial.write(b'\r\n')
        else:
            char = unichr(code)
            if self.settings.echo:
                self.WriteText(char)
            self.serial.write(char.encode('UTF-8', 'replace'))
        event.StopPropagation()

    def WriteText(self, text):
        if False:
            return 10
        if self.settings.unprintable:
            text = ''.join([c if c >= ' ' and c != '\x7f' else unichr(9216 + ord(c)) for c in text])
        self.text_ctrl_output.AppendText(text)

    def OnSerialRead(self, event):
        if False:
            while True:
                i = 10
        'Handle input from the serial port.'
        self.WriteText(event.data.decode('UTF-8', 'replace'))

    def ComPortThread(self):
        if False:
            i = 10
            return i + 15
        '        Thread that handles the incoming traffic. Does the basic input\n        transformation (newlines) and generates an SerialRxEvent\n        '
        while self.alive.is_set():
            b = self.serial.read(self.serial.in_waiting or 1)
            if b:
                if self.settings.newline == NEWLINE_CR:
                    b = b.replace(b'\r', b'\n')
                elif self.settings.newline == NEWLINE_LF:
                    pass
                elif self.settings.newline == NEWLINE_CRLF:
                    b = b.replace(b'\r\n', b'\n')
                wx.PostEvent(self, SerialRxEvent(data=b))

    def OnRTS(self, event):
        if False:
            print('Hello World!')
        self.serial.rts = event.IsChecked()

    def OnDTR(self, event):
        if False:
            i = 10
            return i + 15
        self.serial.dtr = event.IsChecked()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            return 10
        frame_terminal = TerminalFrame(None, -1, '')
        self.SetTopWindow(frame_terminal)
        frame_terminal.Show(True)
        return 1
if __name__ == '__main__':
    app = MyApp(0)
    app.MainLoop()