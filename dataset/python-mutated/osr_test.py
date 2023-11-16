"""Off-screen rendering tests."""
import unittest
import _test_runner
from _common import *
from cefpython3 import cefpython as cef
import sys
g_datauri_data = '\n<!DOCTYPE html>\n<html>\n<head>\n    <style type="text/css">\n    body,html {\n        font-family: Arial;\n        font-size: 11pt;\n    }\n    </style>\n\n    <script>\n    function print(msg) {\n        console.log(msg+" [JS]");\n        msg = msg.replace("ok", "<b style=\'color:green\'>ok</b>");\n        msg = msg.replace("error", "<b style=\'color:red\'>error</b>");\n        document.getElementById("console").innerHTML += msg+"<br>";\n    }\n    function onload_helper() {\n        if (!window.hasOwnProperty("cefpython_version")) {\n            // Sometimes page could finish loading before javascript\n            // bindings are available. Javascript bindings are sent\n            // from the browser process to the renderer process via\n            // IPC messaging and it can take some time (5-10ms). If\n            // the page loads very fast window.onload could execute\n            // before bindings are available.\n            setTimeout(onload_helper, 10);\n            return;\n        }\n        version = cefpython_version\n        print("CEF Python: <b>"+version.version+"</b>");\n        print("Chrome: <b>"+version.chrome_version+"</b>");\n        print("CEF: <b>"+version.cef_version+"</b>");\n        js_code_completed();\n    }\n    window.onload = function() {\n        print("window.onload() ok");\n        onload_helper();\n    }\n    </script>\n</head>\n<body>\n    <!-- FrameSourceVisitor hash = 747ef3e6011b6a61e6b3c6e54bdd2dee -->\n    <h1>Off-screen rendering test</h1>\n    <div id="console"></div>\n    <div id="OnTextSelectionChanged">Test selection.</div>\n</body>\n</html>\n'
g_datauri = cef.GetDataUrl(g_datauri_data)

class OsrTest_IsolatedTest(unittest.TestCase):

    def test_osr(self):
        if False:
            print('Hello World!')
        'Main entry point. All the code must run inside one\n        single test, otherwise strange things happen.'
        print('')
        print('CEF Python {ver}'.format(ver=cef.__version__))
        print('Python {ver}'.format(ver=sys.version[:6]))
        settings = {'debug': False, 'log_severity': cef.LOGSEVERITY_ERROR, 'log_file': '', 'windowless_rendering_enabled': True}
        if not LINUX:
            settings['log_severity'] = cef.LOGSEVERITY_WARNING
        if '--debug' in sys.argv:
            settings['debug'] = True
            settings['log_severity'] = cef.LOGSEVERITY_INFO
        if '--debug-warning' in sys.argv:
            settings['debug'] = True
            settings['log_severity'] = cef.LOGSEVERITY_WARNING
        switches = {'disable-gpu': '', 'disable-gpu-compositing': '', 'enable-begin-frame-scheduling': '', 'disable-surfaces': ''}
        browser_settings = {'windowless_frame_rate': 30}
        cef.Initialize(settings=settings, switches=switches)
        subtest_message('cef.Initialize() ok')
        accessibility_handler = AccessibilityHandler(self)
        cef.SetGlobalClientHandler(accessibility_handler)
        subtest_message('cef.SetGlobalClientHandler() ok')
        global_handler = GlobalHandler(self)
        cef.SetGlobalClientCallback('OnAfterCreated', global_handler._OnAfterCreated)
        subtest_message('cef.SetGlobalClientCallback() ok')
        window_info = cef.WindowInfo()
        window_info.SetAsOffscreen(0)
        browser = cef.CreateBrowserSync(window_info=window_info, settings=browser_settings, url=g_datauri)
        bindings = cef.JavascriptBindings(bindToFrames=False, bindToPopups=False)
        bindings.SetFunction('js_code_completed', js_code_completed)
        bindings.SetProperty('cefpython_version', cef.GetVersion())
        browser.SetJavascriptBindings(bindings)
        subtest_message('browser.SetJavascriptBindings() ok')
        browser.SetAccessibilityState(cef.STATE_ENABLED)
        subtest_message('cef.SetAccessibilityState(STATE_ENABLED) ok')
        client_handlers = [LoadHandler(self, g_datauri), DisplayHandler(self), RenderHandler(self)]
        for handler in client_handlers:
            browser.SetClientHandler(handler)
        browser.SendFocusEvent(True)
        browser.WasResized()
        on_load_end(select_h1_text, browser)
        run_message_loop()
        browser.CloseBrowser(True)
        del browser
        subtest_message('browser.CloseBrowser() ok')
        do_message_loop_work(25)
        check_auto_asserts(self, [] + client_handlers + [global_handler, accessibility_handler])
        cef.Shutdown()
        subtest_message('cef.Shutdown() ok')
        show_test_summary(__file__)
        sys.stdout.flush()

class AccessibilityHandler(object):

    def __init__(self, test_case):
        if False:
            return 10
        self.test_case = test_case
        self.test_for_True = True
        self.javascript_errors_False = False
        self._OnAccessibilityTreeChange_True = False
        self._OnAccessibilityLocationChange_True = False
        self.loadComplete_True = False
        self.layoutComplete_True = False

    def _OnAccessibilityTreeChange(self, value):
        if False:
            print('Hello World!')
        self._OnAccessibilityTreeChange_True = True
        for event in value:
            if 'event_type' in event:
                if event['event_type'] == 'loadComplete':
                    self.test_case.assertFalse(self.loadComplete_True)
                    self.loadComplete_True = True
                elif event['event_type'] == 'layoutComplete':
                    if self.loadComplete_True:
                        self.test_case.assertFalse(self.layoutComplete_True)
                        self.layoutComplete_True = True

    def _OnAccessibilityLocationChange(self, **_):
        if False:
            print('Hello World!')
        self._OnAccessibilityLocationChange_True = True

def select_h1_text(browser):
    if False:
        for i in range(10):
            print('nop')
    browser.SendMouseClickEvent(0, 0, cef.MOUSEBUTTON_LEFT, mouseUp=False, clickCount=1)
    browser.SendMouseMoveEvent(400, 20, mouseLeave=False, modifiers=cef.EVENTFLAG_LEFT_MOUSE_BUTTON)
    browser.SendMouseClickEvent(400, 20, cef.MOUSEBUTTON_LEFT, mouseUp=True, clickCount=1)
    browser.Invalidate(cef.PET_VIEW)
    subtest_message('select_h1_text() ok')

class RenderHandler(object):

    def __init__(self, test_case):
        if False:
            return 10
        self.test_case = test_case
        self.test_for_True = True
        self.GetViewRect_True = False
        self.OnPaint_True = False
        self.OnTextSelectionChanged_True = False

    def GetViewRect(self, rect_out, **_):
        if False:
            return 10
        'Called to retrieve the view rectangle which is relative\n        to screen coordinates. Return True if the rectangle was\n        provided.'
        self.GetViewRect_True = True
        rect_out.extend([0, 0, 800, 600])
        return True

    def OnPaint(self, element_type, paint_buffer, **_):
        if False:
            for i in range(10):
                print('nop')
        'Called when an element should be painted.'
        if element_type == cef.PET_VIEW:
            self.test_case.assertEqual(paint_buffer.width, 800)
            self.test_case.assertEqual(paint_buffer.height, 600)
            if not self.OnPaint_True:
                self.OnPaint_True = True
                subtest_message('RenderHandler.OnPaint: viewport ok')
        else:
            raise Exception('Unsupported element_type in OnPaint')

    def OnTextSelectionChanged(self, selected_text, selected_range, **_):
        if False:
            return 10
        if not self.OnTextSelectionChanged_True:
            self.OnTextSelectionChanged_True = True
            self.test_case.assertEqual(selected_text, '')
            self.test_case.assertEqual(selected_range, [0, 0])
        else:
            self.test_case.assertEqual(selected_text, 'Off-screen rendering test')
if __name__ == '__main__':
    _test_runner.main(os.path.basename(__file__))