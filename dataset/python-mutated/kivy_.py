from cefpython3 import cefpython as cef
import sys
import os
import time
if sys.platform == 'linux':
    import pygtk
    import gtk
    pygtk.require('2.0')
elif sys.platform == 'darwin':
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk
elif sys.platform == 'win32':
    pass
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.base import EventLoop
g_switches = None

class BrowserLayout(BoxLayout):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BrowserLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.browser_widget = CefBrowser()
        layout = BoxLayout()
        layout.size_hint_y = None
        layout.height = 40
        layout.add_widget(Button(text='Back', on_press=self.browser_widget.go_back))
        layout.add_widget(Button(text='Forward', on_press=self.browser_widget.go_forward))
        layout.add_widget(Button(text='Reload', on_press=self.browser_widget.reload))
        layout.add_widget(Button(text='Print', on_press=self.browser_widget.print_page))
        layout.add_widget(Button(text='DevTools', on_press=self.browser_widget.devtools))
        self.add_widget(layout)
        self.add_widget(self.browser_widget)

class CefBrowser(Widget):
    """Represent a browser widget for kivy, which can be used
    like a normal widget."""
    keyboard_mode = 'global'

    def __init__(self, start_url='https://www.google.com/', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CefBrowser, self).__init__(**kwargs)
        for arg in sys.argv:
            if arg.startswith('http://') or arg.startswith('https://'):
                start_url = arg
        self.start_url = start_url
        self.bind(size=self.size_changed)
        self.browser = None
    starting = True

    def size_changed(self, *_):
        if False:
            print('Hello World!')
        'When the height of the cefbrowser widget got changed,\n        create the browser.'
        if self.starting:
            if self.height != 100:
                self.start_cef()
                self.starting = False
        else:
            self.texture = Texture.create(size=self.size, colorfmt='rgba', bufferfmt='ubyte')
            self.texture.flip_vertical()
            with self.canvas:
                Color(1, 1, 1)
                self.rect.size = self.size
            self.browser.WasResized()
    count = 0

    def _message_loop_work(self, *_):
        if False:
            print('Hello World!')
        'Get called every frame.'
        self.count += 1
        cef.MessageLoopWork()
        self.on_mouse_move_emulate()
        if self.count % 2 == 0:
            Clock.schedule_once(self._message_loop_work, 0)
        else:
            Clock.schedule_once(self._message_loop_work, -1)

    def update_rect(self, *_):
        if False:
            for i in range(10):
                print('nop')
        'Get called whenever the texture got updated.\n        => we need to reset the texture for the rectangle\n        '
        self.rect.texture = self.texture

    def start_cef(self):
        if False:
            i = 10
            return i + 15
        'Starts CEF.'
        self.texture = Texture.create(size=self.size, colorfmt='rgba', bufferfmt='ubyte')
        self.texture.flip_vertical()
        with self.canvas:
            Color(1, 1, 1)
            self.rect = Rectangle(size=self.size, texture=self.texture)
        settings = {'browser_subprocess_path': '%s/%s' % (cef.GetModuleDirectory(), 'subprocess'), 'windowless_rendering_enabled': True, 'context_menu': {'enabled': False}, 'external_message_pump': False, 'multi_threaded_message_loop': False}
        if sys.platform == 'linux':
            settings['locales_dir_path'] = cef.GetModuleDirectory() + '/locales'
            settings['resources_dir_path'] = cef.GetModuleDirectory()
        if sys.platform == 'darwin':
            settings['external_message_pump'] = True
        switches = {'disable-surfaces': '', 'disable-gpu': '', 'disable-gpu-compositing': '', 'enable-begin-frame-scheduling': ''}
        browserSettings = {'windowless_frame_rate': 60}
        sys.excepthook = cef.ExceptHook
        cef.WindowUtils.InstallX11ErrorHandlers()
        global g_switches
        g_switches = switches
        cef.Initialize(settings, switches)
        Clock.schedule_once(self._message_loop_work, 0)
        windowInfo = cef.WindowInfo()
        if sys.platform == 'linux':
            gtkwin = gtk.Window()
            gtkwin.realize()
            windowInfo.SetAsOffscreen(gtkwin.window.xid)
        elif sys.platform == 'darwin' or sys.platform == 'win32':
            windowInfo.SetAsOffscreen(0)
        self.browser = cef.CreateBrowserSync(windowInfo, browserSettings, navigateUrl=self.start_url)
        self.browser.SendFocusEvent(True)
        self._client_handler = ClientHandler(self)
        self.browser.SetClientHandler(self._client_handler)
        self.set_js_bindings()
        self.browser.WasResized()
        self.browser.SetUserData('browserWidget', self)
        if self.keyboard_mode == 'global':
            self.request_keyboard()
    _client_handler = None
    _js_bindings = None

    def set_js_bindings(self):
        if False:
            return 10
        if not self._js_bindings:
            self._js_bindings = cef.JavascriptBindings(bindToFrames=True, bindToPopups=True)
            self._js_bindings.SetFunction('__kivy__request_keyboard', self.request_keyboard)
            self._js_bindings.SetFunction('__kivy__release_keyboard', self.release_keyboard)
        self.browser.SetJavascriptBindings(self._js_bindings)

    def change_url(self, *_):
        if False:
            for i in range(10):
                print('nop')
        self.browser.StopLoad()
        self.browser.GetMainFrame().ExecuteJavascript("window.location='http://www.youtube.com/'")
    _keyboard = None

    def request_keyboard(self):
        if False:
            return 10
        print('[kivy_.py] request_keyboard()')
        self._keyboard = EventLoop.window.request_keyboard(self.release_keyboard, self)
        self._keyboard.bind(on_key_down=self.on_key_down)
        self._keyboard.bind(on_key_up=self.on_key_up)
        self.is_shift1 = False
        self.is_shift2 = False
        self.is_ctrl1 = False
        self.is_ctrl2 = False
        self.is_alt1 = False
        self.is_alt2 = False
        self.browser.SendFocusEvent(True)

    def release_keyboard(self):
        if False:
            print('Hello World!')
        self.is_shift1 = False
        self.is_shift2 = False
        self.is_ctrl1 = False
        self.is_ctrl2 = False
        self.is_alt1 = False
        self.is_alt2 = False
        if not self._keyboard:
            return
        print('[kivy_.py] release_keyboard()')
        self._keyboard.unbind(on_key_down=self.on_key_down)
        self._keyboard.unbind(on_key_up=self.on_key_up)
        self._keyboard.release()
    is_shift1 = False
    is_shift2 = False
    is_ctrl1 = False
    is_ctrl2 = False
    is_alt1 = False
    is_alt2 = False

    def on_key_down(self, _, key, text, modifiers):
        if False:
            while True:
                i = 10
        if text:
            pass
        if key[0] == -1:
            return
        if key[0] == 27:
            self.browser.GetFocusedFrame().ExecuteJavascript('__kivy__on_escape()')
            return
        if key[0] in (306, 305):
            modifiers.append('ctrl')
        cef_modifiers = cef.EVENTFLAG_NONE
        if 'shift' in modifiers:
            cef_modifiers |= cef.EVENTFLAG_SHIFT_DOWN
        if 'ctrl' in modifiers:
            cef_modifiers |= cef.EVENTFLAG_CONTROL_DOWN
        if 'alt' in modifiers:
            cef_modifiers |= cef.EVENTFLAG_ALT_DOWN
        if 'capslock' in modifiers:
            cef_modifiers |= cef.EVENTFLAG_CAPS_LOCK_ON
        keycode = self.get_windows_key_code(key[0])
        charcode = key[0]
        if text:
            charcode = ord(text)
        keyEvent = {'type': cef.KEYEVENT_RAWKEYDOWN, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
        self.browser.SendKeyEvent(keyEvent)
        if text:
            keyEvent = {'type': cef.KEYEVENT_CHAR, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
            self.browser.SendKeyEvent(keyEvent)
        if key[0] == 304:
            self.is_shift1 = True
        elif key[0] == 303:
            self.is_shift2 = True
        elif key[0] == 306:
            self.is_ctrl1 = True
        elif key[0] == 305:
            self.is_ctrl2 = True
        elif key[0] == 308:
            self.is_alt1 = True
        elif key[0] == 313:
            self.is_alt2 = True

    def on_key_up(self, _, key):
        if False:
            i = 10
            return i + 15
        if key[0] == -1:
            return
        cef_modifiers = cef.EVENTFLAG_NONE
        if self.is_shift1 or self.is_shift2:
            cef_modifiers |= cef.EVENTFLAG_SHIFT_DOWN
        if self.is_ctrl1 or self.is_ctrl2:
            cef_modifiers |= cef.EVENTFLAG_CONTROL_DOWN
        if self.is_alt1:
            cef_modifiers |= cef.EVENTFLAG_ALT_DOWN
        keycode = self.get_windows_key_code(key[0])
        charcode = key[0]
        keyEvent = {'type': cef.KEYEVENT_KEYUP, 'windows_key_code': keycode, 'character': charcode, 'unmodified_character': charcode, 'modifiers': cef_modifiers}
        self.browser.SendKeyEvent(keyEvent)
        if key[0] == 304:
            self.is_shift1 = False
        elif key[0] == 303:
            self.is_shift2 = False
        elif key[0] == 306:
            self.is_ctrl1 = False
        elif key[0] == 305:
            self.is_ctrl2 = False
        elif key[0] == 308:
            self.is_alt1 = False
        elif key[0] == 313:
            self.is_alt2 = False

    def get_windows_key_code(self, kivycode):
        if False:
            return 10
        cefcode = kivycode
        if 97 <= kivycode <= 122:
            cefcode = kivycode - 32
        other_keys_map = {'27': 27, '282': 112, '283': 113, '284': 114, '285': 115, '286': 116, '287': 117, '288': 118, '289': 119, '290': 120, '291': 121, '292': 122, '293': 123, '9': 9, '304': 16, '303': 16, '306': 17, '305': 17, '308': 18, '313': 225, '8': 8, '13': 13, '316': 42, '302': 145, '19': 19, '277': 45, '127': 46, '278': 36, '279': 35, '280': 33, '281': 34, '276': 37, '273': 38, '275': 39, '274': 40, '96': 192, '45': 189, '61': 187, '91': 219, '93': 221, '92': 220, '311': 91, '59': 186, '39': 222, '44': 188, '46': 190, '47': 91, '319': 0}
        if str(kivycode) in other_keys_map:
            cefcode = other_keys_map[str(kivycode)]
        return cefcode

    def go_forward(self, *_):
        if False:
            print('Hello World!')
        'Going to forward in browser history.'
        print('go forward')
        self.browser.GoForward()

    def go_back(self, *_):
        if False:
            print('Hello World!')
        'Going back in browser history.'
        print('go back')
        self.browser.GoBack()

    def reload(self, *_):
        if False:
            for i in range(10):
                print('nop')
        self.browser.Reload()

    def print_page(self, *_):
        if False:
            i = 10
            return i + 15
        self.browser.Print()

    def devtools(self, *_):
        if False:
            i = 10
            return i + 15
        if 'enable-begin-frame-scheduling' in g_switches:
            text = "To enable DevTools you need to remove the\n'enable-begin-frame-scheduling' switch that\nis passed to cef.Initialize(). See also\ncomment in CefBrowser.devtools()."
            popup = Popup(title='DevTools INFO', content=Label(text=text), size_hint=(None, None), size=(400, 400))
            popup.open()
        else:
            self.browser.ShowDevTools()
    is_mouse_down = False
    is_drag = False
    is_drag_leave = False
    drag_data = None
    current_drag_operation = cef.DRAG_OPERATION_NONE

    def on_touch_down(self, touch, *kwargs):
        if False:
            return 10
        if 'button' in touch.profile:
            if touch.button in ['scrollup', 'scrolldown']:
                return
        if not self.collide_point(*touch.pos):
            return
        touch.grab(self)
        y = self.height - touch.pos[1]
        if touch.is_double_tap:
            self.browser.SendMouseClickEvent(touch.x, y, cef.MOUSEBUTTON_RIGHT, mouseUp=False, clickCount=1)
            self.browser.SendMouseClickEvent(touch.x, y, cef.MOUSEBUTTON_RIGHT, mouseUp=True, clickCount=1)
        else:
            self.browser.SendMouseClickEvent(touch.x, y, cef.MOUSEBUTTON_LEFT, mouseUp=False, clickCount=1)
            self.is_mouse_down = True

    def on_touch_up(self, touch, *kwargs):
        if False:
            print('Hello World!')
        if 'button' in touch.profile:
            if touch.button in ['scrollup', 'scrolldown']:
                x = touch.x
                y = self.height - touch.pos[1]
                deltaY = -40 if 'scrollup' == touch.button else 40
                self.browser.SendMouseWheelEvent(x, y, deltaX=0, deltaY=deltaY)
                return
        if touch.grab_current is not self:
            return
        y = self.height - touch.pos[1]
        self.browser.SendMouseClickEvent(touch.x, y, cef.MOUSEBUTTON_LEFT, mouseUp=True, clickCount=1)
        self.is_mouse_down = False
        if self.is_drag:
            if self.is_drag_leave or not self.is_inside_web_view(touch.x, y):
                x = touch.x
                if x == 0:
                    x = -1
                if x == self.width - 1:
                    x = self.width
                if y == 0:
                    y = -1
                if y == self.height - 1:
                    y = self.height
                print('[kivy_.py] ~~ DragSourceEndedAt')
                print('[kivy_.py] ~~ current_drag_operation=%s' % self.current_drag_operation)
                self.browser.DragSourceEndedAt(x, y, self.current_drag_operation)
                self.drag_ended()
            else:
                print('[kivy_.py] ~~ DragTargetDrop')
                print('[kivy_.py] ~~ DragSourceEndedAt')
                print('[kivy_.py] ~~ current_drag_operation=%s' % self.current_drag_operation)
                self.browser.DragTargetDrop(touch.x, y)
                self.browser.DragSourceEndedAt(touch.x, y, self.current_drag_operation)
                self.drag_ended()
        touch.ungrab(self)
    last_mouse_pos = None

    def on_mouse_move_emulate(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self.get_root_window(), 'mouse_pos'):
            return
        mouse_pos = self.get_root_window().mouse_pos
        if self.last_mouse_pos == mouse_pos:
            return
        self.last_mouse_pos = mouse_pos
        x = mouse_pos[0]
        y = int(mouse_pos[1] - self.height)
        if x >= 0 >= y:
            y = abs(y)
            if not self.is_mouse_down and (not self.is_drag):
                self.browser.SendMouseMoveEvent(x, y, mouseLeave=False)

    def on_touch_move(self, touch, *kwargs):
        if False:
            for i in range(10):
                print('nop')
        if touch.grab_current is not self:
            return
        y = self.height - touch.pos[1]
        modifiers = cef.EVENTFLAG_LEFT_MOUSE_BUTTON
        self.browser.SendMouseMoveEvent(touch.x, y, mouseLeave=False, modifiers=modifiers)
        if self.is_drag:
            if self.is_inside_web_view(touch.x, y):
                if self.is_drag_leave:
                    print('[kivy_.py] ~~ DragTargetDragEnter')
                    self.browser.DragTargetDragEnter(self.drag_data, touch.x, y, cef.DRAG_OPERATION_EVERY)
                    self.is_drag_leave = False
                print('[kivy_.py] ~~ DragTargetDragOver')
                self.browser.DragTargetDragOver(touch.x, y, cef.DRAG_OPERATION_EVERY)
                self.update_drag_icon(touch.x, y)
            elif not self.is_drag_leave:
                self.is_drag_leave = True
                print('[kivy_.py] ~~ DragTargetDragLeave')
                self.browser.DragTargetDragLeave()

    def is_inside_web_view(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        if 0 < x < self.width - 1 and 0 < y < self.height - 1:
            return True
        return False

    def drag_ended(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_drag = False
        self.is_drag_leave = False
        del self.drag_data
        self.current_drag_operation = cef.DRAG_OPERATION_NONE
        self.update_drag_icon(None, None)
        print('[kivy_.py] ~~ DragSourceSystemDragEnded')
        self.browser.DragSourceSystemDragEnded()
    drag_icon = None

    def update_drag_icon(self, x, y):
        if False:
            print('Hello World!')
        if self.is_drag:
            if self.drag_icon:
                self.drag_icon.pos = self.flip_pos_vertical(x, y)
            else:
                image = self.drag_data.GetImage()
                width = image.GetWidth()
                height = image.GetHeight()
                abuffer = image.GetAsBitmap(1.0, cef.CEF_COLOR_TYPE_BGRA_8888, cef.CEF_ALPHA_TYPE_PREMULTIPLIED)
                texture = Texture.create(size=(width, height))
                texture.blit_buffer(abuffer, colorfmt='bgra', bufferfmt='ubyte')
                texture.flip_vertical()
                self.drag_icon = Rectangle(texture=texture, pos=self.flip_pos_vertical(x, y), size=(width, height))
                self.canvas.add(self.drag_icon)
        elif self.drag_icon:
            self.canvas.remove(self.drag_icon)
            del self.drag_icon

    def flip_pos_vertical(self, x, y):
        if False:
            return 10
        half = self.height / 2
        if y > half:
            y = half - (y - half)
        elif y < half:
            y = half + (half - y)
        y -= 20
        x -= 20
        return (x, y)

class ClientHandler:

    def __init__(self, browserWidget):
        if False:
            for i in range(10):
                print('nop')
        self.browserWidget = browserWidget

    def _fix_select_boxes(self, frame):
        if False:
            print('Hello World!')
        print('[kivy_.py] _fix_select_boxes()')
        resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kivy-select-boxes')
        if not os.path.exists(resources_dir):
            print("[kivy_.py] The kivy-select-boxes directory does not exist, select boxes fix won't be applied.")
            return
        js_file = os.path.join(resources_dir, 'kivy-selectBox.js')
        js_content = ''
        with open(js_file, 'r') as myfile:
            js_content = myfile.read()
        css_file = os.path.join(resources_dir, 'kivy-selectBox.css')
        css_content = ''
        with open(css_file, 'r') as myfile:
            css_content = myfile.read()
        css_content = css_content.replace('\r', '')
        css_content = css_content.replace('\n', '')
        jsCode = '\n            %(js_content)s\n            var __kivy_temp_head = document.getElementsByTagName(\'head\')[0];\n            var __kivy_temp_style = document.createElement(\'style\');\n            __kivy_temp_style.type = \'text/css\';\n            __kivy_temp_style.appendChild(document.createTextNode("%(css_content)s"));\n            __kivy_temp_head.appendChild(__kivy_temp_style);\n        ' % locals()
        frame.ExecuteJavascript(jsCode, 'kivy_.py > ClientHandler > OnLoadStart > _fix_select_boxes()')

    def OnLoadStart(self, browser, frame, **_):
        if False:
            i = 10
            return i + 15
        self.load_start_time = time.time()
        self._fix_select_boxes(frame)
        browserWidget = browser.GetUserData('browserWidget')
        if browserWidget and browserWidget.keyboard_mode == 'local':
            print('[kivy_.py] OnLoadStart(): injecting focus listeners for text controls')
            jsCode = '\n                var __kivy__keyboard_requested = false;\n                function __kivy__keyboard_interval() {\n                    var element = document.activeElement;\n                    if (!element) {\n                        return;\n                    }\n                    var tag = element.tagName;\n                    var type = element.type;\n                    if (tag == "INPUT" && (type == "" || type == "text"\n                            || type == "password") || tag == "TEXTAREA") {\n                        if (!__kivy__keyboard_requested) {\n                            __kivy__request_keyboard();\n                            __kivy__keyboard_requested = true;\n                        }\n                        return;\n                    }\n                    if (__kivy__keyboard_requested) {\n                        __kivy__release_keyboard();\n                        __kivy__keyboard_requested = false;\n                    }\n                }\n                function __kivy__on_escape() {\n                    if (document.activeElement) {\n                        document.activeElement.blur();\n                    }\n                    if (__kivy__keyboard_requested) {\n                        __kivy__release_keyboard();\n                        __kivy__keyboard_requested = false;\n                    }\n                }\n                setInterval(__kivy__keyboard_interval, 100);\n            '
            frame.ExecuteJavascript(jsCode, 'kivy_.py > ClientHandler > OnLoadStart')

    def OnLoadEnd(self, browser, **_):
        if False:
            i = 10
            return i + 15
        browserWidget = browser.GetUserData('browserWidget')
        if browserWidget and browserWidget.keyboard_mode == 'global':
            browser.SendFocusEvent(True)

    def OnLoadingStateChange(self, is_loading, **_):
        if False:
            return 10
        print('[kivy_.py] OnLoadingStateChange: isLoading = %s' % is_loading)
        if self.load_start_time:
            print('[kivy_.py] OnLoadingStateChange: load time = {time}'.format(time=time.time() - self.load_start_time))
            self.load_start_time = None

    def OnPaint(self, element_type, paint_buffer, **_):
        if False:
            return 10
        if element_type != cef.PET_VIEW:
            print("Popups aren't implemented yet")
            return
        if 'fps' in sys.argv:
            if not hasattr(self, 'last_paints'):
                self.last_paints = []
            self.last_paints.append(time.time())
            while len(self.last_paints) > 30:
                self.last_paints.pop(0)
            if len(self.last_paints) > 1:
                fps = len(self.last_paints) / (self.last_paints[-1] - self.last_paints[0])
                print('[kivy_.py] FPS={fps}'.format(fps=fps))
        paint_buffer = paint_buffer.GetString(mode='bgra', origin='top-left')
        self.browserWidget.texture.blit_buffer(paint_buffer, colorfmt='bgra', bufferfmt='ubyte')
        self.browserWidget.update_rect()
        return True

    def GetViewRect(self, rect_out, **_):
        if False:
            print('Hello World!')
        (width, height) = self.browserWidget.texture.size
        rect_out.append(0)
        rect_out.append(0)
        rect_out.append(width)
        rect_out.append(height)
        return True
    '\n\n    def GetRootScreenRect(self, rect_out, **_):\n        width, height = self.browserWidget.texture.size\n        rect_out.append(0)\n        rect_out.append(0)\n        rect_out.append(width)\n        rect_out.append(height)\n        return True\n\n\n    def GetScreenRect(self, rect_out, **_):\n        width, height = self.browserWidget.texture.size\n        rect_out.append(0)\n        rect_out.append(0)\n        rect_out.append(width)\n        rect_out.append(height)\n        return True\n\n\n    def GetScreenPoint(self, screen_coordinates_out, **kwargs):\n        screen_coordinates_out.append(view_x)\n        screen_coordinates_out.append(view_y)\n        return True\n\n    '

    def OnJavascriptDialog(self, suppress_message_out, **_):
        if False:
            print('Hello World!')
        suppress_message_out[0] = True
        return False

    def OnBeforeUnloadJavascriptDialog(self, callback, **_):
        if False:
            for i in range(10):
                print('nop')
        callback.Continue(allow=True, userInput='')
        return True

    def StartDragging(self, drag_data, x, y, **_):
        if False:
            while True:
                i = 10
        print('[kivy_.py] ~~ StartDragging')
        print('[kivy_.py] ~~ DragTargetDragEnter')
        self.browserWidget.browser.DragTargetDragEnter(drag_data, x, y, cef.DRAG_OPERATION_EVERY)
        self.browserWidget.is_drag = True
        self.browserWidget.is_drag_leave = False
        self.browserWidget.drag_data = drag_data
        self.browserWidget.current_drag_operation = cef.DRAG_OPERATION_NONE
        self.browserWidget.update_drag_icon(x, y)
        return True

    def UpdateDragCursor(self, **kwargs):
        if False:
            return 10
        self.browserWidget.current_drag_operation = kwargs['operation']

class CefBrowserApp(App):

    def build(self):
        if False:
            while True:
                i = 10
        self.layout = BrowserLayout()
        return self.layout

    def on_stop(self):
        if False:
            i = 10
            return i + 15
        self.layout.browser_widget.browser.CloseBrowser(True)
        del self.layout.browser_widget.browser
if __name__ == '__main__':
    CefBrowserApp().run()
    cef.Shutdown()