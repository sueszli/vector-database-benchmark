""" This module implements a Panda3D window that can be embedded
within a wx.Frame.  The window itself is either an embedded window,
which is a wx.Window with the Panda window attached to it, or it is a
wxgl.GLCanvas, with Panda directed to draw into it, depending on the
platform.  In either case, you may simply embed this window into a
wx.Frame of your choosing, using sizers or whatever you like. """
import wx
import platform
try:
    import wx.glcanvas as wxgl
except ImportError:
    wxgl = None
from panda3d.core import CallbackGraphicsWindow, ConfigVariableBool, FrameBufferProperties, KeyboardButton, MouseButton, WindowProperties
__all__ = ['WxPandaWindow']

class EmbeddedPandaWindow(wx.Window):
    """ This class implements a Panda3D window that is directly
    embedded within the frame.  It is fully supported on Windows,
    partially supported on Linux, and not at all on OSX. """

    def __init__(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        gsg = None
        if 'gsg' in kw:
            gsg = kw['gsg']
            del kw['gsg']
        base.startWx()
        wx.Window.__init__(self, *args, **kw)
        wp = WindowProperties.getDefault()
        if platform.system() != 'Darwin':
            try:
                wp.setParentWindow(self.GetHandle())
            except OverflowError:
                wp.setParentWindow(self.GetHandle() & 4294967295)
        self.win = base.openWindow(props=wp, gsg=gsg, type='onscreen', unexposedDraw=False)
        self.Bind(wx.EVT_SIZE, self.onSize)
        self.Bind(wx.EVT_CLOSE, self.__closeEvent)

    def __closeEvent(self, event):
        if False:
            while True:
                i = 10
        self.cleanup()
        event.Skip()

    def cleanup(self):
        if False:
            print('Hello World!')
        " Parent windows should call cleanup() to clean up the\n        wxPandaWindow explicitly (since we can't catch EVT_CLOSE\n        directly). "
        if self.win:
            base.closeWindow(self.win)
            self.win = None

    def onSize(self, event):
        if False:
            print('Hello World!')
        if self.win:
            wp = WindowProperties()
            wp.setOrigin(0, 0)
            wp.setSize(*self.GetClientSize())
            self.win.requestProperties(wp)
        event.Skip()
if not hasattr(wxgl, 'GLCanvas'):
    OpenGLPandaWindow = None
else:

    class OpenGLPandaWindow(wxgl.GLCanvas):
        """ This class implements a Panda3D "window" that actually draws
        within the wx GLCanvas object.  It is supported whenever OpenGL is
        Panda's rendering engine, and GLCanvas is available in wx. """
        removeCallbackWindow = ConfigVariableBool('remove-callback-window', True)
        Keymap = {wx.WXK_BACK: KeyboardButton.backspace(), wx.WXK_TAB: KeyboardButton.tab(), wx.WXK_RETURN: KeyboardButton.enter(), wx.WXK_ESCAPE: KeyboardButton.escape(), wx.WXK_DELETE: KeyboardButton._del(), wx.WXK_SHIFT: KeyboardButton.shift(), wx.WXK_ALT: KeyboardButton.alt(), wx.WXK_CONTROL: KeyboardButton.control(), wx.WXK_MENU: KeyboardButton.meta(), wx.WXK_PAUSE: KeyboardButton.pause(), wx.WXK_END: KeyboardButton.end(), wx.WXK_HOME: KeyboardButton.home(), wx.WXK_LEFT: KeyboardButton.left(), wx.WXK_UP: KeyboardButton.up(), wx.WXK_RIGHT: KeyboardButton.right(), wx.WXK_DOWN: KeyboardButton.down(), wx.WXK_PRINT: KeyboardButton.printScreen(), wx.WXK_INSERT: KeyboardButton.insert(), wx.WXK_F1: KeyboardButton.f1(), wx.WXK_F2: KeyboardButton.f2(), wx.WXK_F3: KeyboardButton.f3(), wx.WXK_F4: KeyboardButton.f4(), wx.WXK_F5: KeyboardButton.f5(), wx.WXK_F6: KeyboardButton.f6(), wx.WXK_F7: KeyboardButton.f7(), wx.WXK_F8: KeyboardButton.f8(), wx.WXK_F9: KeyboardButton.f9(), wx.WXK_F10: KeyboardButton.f10(), wx.WXK_F11: KeyboardButton.f11(), wx.WXK_F12: KeyboardButton.f12(), wx.WXK_NUMLOCK: KeyboardButton.numLock(), wx.WXK_SCROLL: KeyboardButton.scrollLock(), wx.WXK_PAGEUP: KeyboardButton.pageUp(), wx.WXK_PAGEDOWN: KeyboardButton.pageDown(), wx.WXK_COMMAND: KeyboardButton.meta()}

        def __init__(self, *args, **kw):
            if False:
                print('Hello World!')
            gsg = None
            if 'gsg' in kw:
                gsg = kw['gsg']
                del kw['gsg']
            fbprops = kw.get('fbprops', None)
            if fbprops is None:
                fbprops = FrameBufferProperties.getDefault()
            attribList = kw.get('attribList', None)
            if attribList is None:
                attribList = [wxgl.WX_GL_RGBA, True, wxgl.WX_GL_LEVEL, 0]
                if not fbprops.isSingleBuffered():
                    attribList.append(wxgl.WX_GL_DOUBLEBUFFER)
                    attribList.append(True)
                if fbprops.getDepthBits() > 0:
                    attribList.append(wxgl.WX_GL_DEPTH_SIZE)
                    if fbprops.getDepthBits() <= 16:
                        attribList.append(16)
                    elif fbprops.getDepthBits() <= 24:
                        attribList.append(24)
                    else:
                        attribList.append(32)
                kw['attribList'] = attribList
            base.startWx()
            wxgl.GLCanvas.__init__(self, *args, **kw)
            self.visible = False
            gsg = None
            callbackWindowDict = {'Events': self.__eventsCallback, 'Properties': self.__propertiesCallback, 'Render': self.__renderCallback}
            if not base.pipe:
                base.makeDefaultPipe()
            pipe = base.pipe
            if pipe.getInterfaceName() != 'OpenGL':
                base.makeAllPipes()
                for pipe in base.pipeList:
                    if pipe.getInterfaceName() == 'OpenGL':
                        break
            if pipe.getInterfaceName() != 'OpenGL':
                raise Exception("Couldn't get an OpenGL pipe.")
            self.win = base.openWindow(callbackWindowDict=callbackWindowDict, pipe=pipe, gsg=gsg, type='onscreen')
            self.hasCapture = False
            self.inputDevice = None
            if hasattr(self.win, 'getInputDevice'):
                self.inputDevice = self.win.getInputDevice(0)
            self.Bind(wx.EVT_SIZE, self.onSize)
            self.Bind(wx.EVT_PAINT, self.onPaint)
            self.Bind(wx.EVT_IDLE, self.onIdle)
            if self.inputDevice:
                self.Bind(wx.EVT_LEFT_DOWN, lambda event: self.__buttonDown(MouseButton.one()))
                self.Bind(wx.EVT_LEFT_UP, lambda event: self.__buttonUp(MouseButton.one()))
                self.Bind(wx.EVT_MIDDLE_DOWN, lambda event: self.__buttonDown(MouseButton.two()))
                self.Bind(wx.EVT_MIDDLE_UP, lambda event: self.__buttonUp(MouseButton.two()))
                self.Bind(wx.EVT_RIGHT_DOWN, lambda event: self.__buttonDown(MouseButton.three()))
                self.Bind(wx.EVT_RIGHT_UP, lambda event: self.__buttonUp(MouseButton.three()))
                self.Bind(wx.EVT_MOTION, self.__mouseMotion)
                self.Bind(wx.EVT_MOUSEWHEEL, self.__mouseWheel)
                self.Bind(wx.EVT_LEAVE_WINDOW, self.__mouseLeaveWindow)
                self.Bind(wx.EVT_KEY_DOWN, self.__keyDown)
                self.Bind(wx.EVT_KEY_UP, self.__keyUp)
                self.Bind(wx.EVT_CHAR, self.__keystroke)
            self.Bind(wx.EVT_CLOSE, self.__closeEvent)

        def __closeEvent(self, event):
            if False:
                print('Hello World!')
            self.cleanup()
            event.Skip()

        def cleanup(self):
            if False:
                while True:
                    i = 10
            " Parent windows should call cleanup() to clean up the\n            wxPandaWindow explicitly (since we can't catch EVT_CLOSE\n            directly). "
            if self.win:
                self.win.clearEventsCallback()
                self.win.clearPropertiesCallback()
                self.win.clearRenderCallback()
                base.closeWindow(self.win, removeWindow=self.removeCallbackWindow)
                self.win = None

        def __buttonDown(self, button):
            if False:
                i = 10
                return i + 15
            self.SetFocus()
            if not self.hasCapture:
                self.CaptureMouse()
                self.hasCapture = True
            self.inputDevice.buttonDown(button)

        def __buttonUp(self, button):
            if False:
                for i in range(10):
                    print('nop')
            if self.hasCapture:
                self.ReleaseMouse()
                self.hasCapture = False
            self.inputDevice.buttonUp(button)

        def __mouseMotion(self, event):
            if False:
                return 10
            self.inputDevice.setPointerInWindow(*event.GetPosition())

        def __mouseWheel(self, event):
            if False:
                for i in range(10):
                    print('nop')
            amount = event.GetWheelRotation()
            if amount > 0.0:
                self.inputDevice.buttonDown(MouseButton.wheelUp())
                self.inputDevice.buttonUp(MouseButton.wheelUp())
            elif amount < 0.0:
                self.inputDevice.buttonDown(MouseButton.wheelDown())
                self.inputDevice.buttonUp(MouseButton.wheelDown())

        def __mouseLeaveWindow(self, event):
            if False:
                while True:
                    i = 10
            self.inputDevice.setPointerOutOfWindow()

        def __keyDown(self, event):
            if False:
                i = 10
                return i + 15
            key = self.__getkey(event)
            if key:
                self.inputDevice.buttonDown(key)

        def __keyUp(self, event):
            if False:
                i = 10
                return i + 15
            key = self.__getkey(event)
            if key:
                self.inputDevice.buttonUp(key)

        def __getkey(self, event):
            if False:
                print('Hello World!')
            code = event.GetKeyCode()
            key = self.Keymap.get(code, None)
            if key is not None:
                return key
            if code >= 65 and code <= 90:
                return KeyboardButton.asciiKey(code + 32)
            if code >= 32 and code <= 128:
                return KeyboardButton.asciiKey(code)
            return None

        def __keystroke(self, event):
            if False:
                i = 10
                return i + 15
            if hasattr(event, 'GetUnicodeKey'):
                code = event.GetUnicodeKey()
            else:
                code = event.GetKeyCode()
                if code < 32 or code >= 128:
                    return
            self.inputDevice.keystroke(code)

        def __eventsCallback(self, data):
            if False:
                return 10
            data.upcall()

        def __propertiesCallback(self, data):
            if False:
                while True:
                    i = 10
            data.upcall()

        def __renderCallback(self, data):
            if False:
                while True:
                    i = 10
            cbType = data.getCallbackType()
            if cbType == CallbackGraphicsWindow.RCTBeginFrame:
                if not self.visible:
                    data.setRenderFlag(False)
                    return
                self.SetCurrent()
                return
            elif cbType == CallbackGraphicsWindow.RCTEndFlip:
                self.SwapBuffers()
                self.Refresh()
                self.visible = False
            data.upcall()

        def onSize(self, event):
            if False:
                for i in range(10):
                    print('nop')
            wp = WindowProperties()
            wp.setSize(*self.GetClientSize())
            self.win.requestProperties(wp)
            wx.WakeUpIdle()
            event.Skip()

        def onPaint(self, event):
            if False:
                for i in range(10):
                    print('nop')
            ' This is called whenever we get the first paint event,\n            at which point we can conclude that the window has\n            actually been manifested onscreen.  (In X11, there appears\n            to be no way to know this otherwise.) '
            self.visible = True

        def onIdle(self, event):
            if False:
                while True:
                    i = 10
            if self.win:
                size = None
                properties = self.win.getProperties()
                if properties.hasSize():
                    size = (properties.getXSize(), properties.getYSize())
                if tuple(self.GetClientSize()) != size:
                    wp = WindowProperties()
                    wp.setSize(*self.GetClientSize())
                    self.win.requestProperties(wp)
            event.Skip()
WxPandaWindow = None
if platform.system() == 'Darwin' or platform.system() == 'Linux':
    WxPandaWindow = OpenGLPandaWindow
if not WxPandaWindow:
    WxPandaWindow = EmbeddedPandaWindow