"""
Implement RequestHandler.CanGetCookies and CanSetCookie
to block or allow cookies over network requests.
"""
from cefpython3 import cefpython as cef

def main():
    if False:
        i = 10
        return i + 15
    cef.Initialize()
    browser = cef.CreateBrowserSync(url='http://www.html-kit.com/tools/cookietester/', window_title='Network cookies')
    browser.SetClientHandler(RequestHandler())
    cef.MessageLoop()
    del browser
    cef.Shutdown()

class RequestHandler(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.getcount = 0
        self.setcount = 0

    def CanGetCookies(self, frame, request, **_):
        if False:
            for i in range(10):
                print('nop')
        if frame.IsMain():
            self.getcount += 1
            print('-- CanGetCookies #' + str(self.getcount))
            print('url=' + request.GetUrl()[0:80])
            print('')
        return True

    def CanSetCookie(self, frame, request, cookie, **_):
        if False:
            print('Hello World!')
        if frame.IsMain():
            self.setcount += 1
            print('-- CanSetCookie @' + str(self.setcount))
            print('url=' + request.GetUrl()[0:80])
            print('Name=' + cookie.GetName())
            print('Value=' + cookie.GetValue())
            print('')
        return True
if __name__ == '__main__':
    main()