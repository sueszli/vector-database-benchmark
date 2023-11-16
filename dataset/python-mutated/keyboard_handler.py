from cefpython3 import cefpython as cef

def main():
    if False:
        i = 10
        return i + 15
    cef.Initialize()
    browser = cef.CreateBrowserSync(url='https://www.google.com/', window_title='Keyboard Handler')
    browser.SetClientHandler(KeyboardHandler())
    cef.MessageLoop()
    del browser
    cef.Shutdown()

class KeyboardHandler(object):

    def OnKeyEvent(self, browser, event, event_handle, **_):
        if False:
            i = 10
            return i + 15
        print('OnKeyEvent: ' + str(event))
if __name__ == '__main__':
    main()