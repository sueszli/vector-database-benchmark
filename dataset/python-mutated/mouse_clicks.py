from cefpython3 import cefpython as cef

def main():
    if False:
        while True:
            i = 10
    cef.Initialize()
    browser = cef.CreateBrowserSync(url='data:text/html,<h1>Mouse clicks snippet</h1>This text will be selected after one second.<br>This text will be selected after two seconds.', window_title='Mouse clicks')
    browser.SetClientHandler(LifespanHandler())
    cef.MessageLoop()
    del browser
    cef.Shutdown()

def click_after_1_second(browser):
    if False:
        for i in range(10):
            print('nop')
    print('Click after 1 second')
    browser.SendMouseMoveEvent(0, 70, False, 0)
    browser.SendMouseClickEvent(0, 70, cef.MOUSEBUTTON_LEFT, False, 1)
    browser.SendMouseMoveEvent(400, 80, False, cef.EVENTFLAG_LEFT_MOUSE_BUTTON)
    browser.SendMouseClickEvent(400, 80, cef.MOUSEBUTTON_LEFT, True, 1)
    cef.PostDelayedTask(cef.TID_UI, 1000, click_after_2_seconds, browser)

def click_after_2_seconds(browser):
    if False:
        i = 10
        return i + 15
    print('Click after 2 seconds')
    browser.SendMouseMoveEvent(0, 90, False, 0)
    browser.SendMouseClickEvent(0, 90, cef.MOUSEBUTTON_LEFT, False, 1)
    browser.SendMouseMoveEvent(400, 99, False, cef.EVENTFLAG_LEFT_MOUSE_BUTTON)
    browser.SendMouseClickEvent(400, 99, cef.MOUSEBUTTON_LEFT, True, 1)
    cef.PostDelayedTask(cef.TID_UI, 1000, click_after_1_second, browser)

class LifespanHandler(object):

    def OnLoadEnd(self, browser, **_):
        if False:
            return 10
        print('Page loading is complete')
        cef.PostDelayedTask(cef.TID_UI, 1000, click_after_1_second, browser)
if __name__ == '__main__':
    main()