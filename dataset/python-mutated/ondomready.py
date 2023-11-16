"""
Execute custom Python code on a web page as soon as DOM is ready.
Implements a custom "_OnDomReady" event in the LoadHandler object.
"""
from cefpython3 import cefpython as cef

def main():
    if False:
        for i in range(10):
            print('nop')
    cef.Initialize()
    browser = cef.CreateBrowserSync(url='https://www.google.com/', window_title='_OnDomReady event')
    load_handler = LoadHandler(browser)
    browser.SetClientHandler(load_handler)
    bindings = cef.JavascriptBindings()
    bindings.SetFunction('LoadHandler_OnDomReady', load_handler['_OnDomReady'])
    browser.SetJavascriptBindings(bindings)
    cef.MessageLoop()
    del load_handler
    del browser
    cef.Shutdown()

class LoadHandler(object):

    def __init__(self, browser):
        if False:
            print('Hello World!')
        self.browser = browser

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return getattr(self, key)

    def OnLoadStart(self, browser, **_):
        if False:
            for i in range(10):
                print('nop')
        browser.ExecuteJavascript('\n            if (document.readyState === "complete") {\n                LoadHandler_OnDomReady();\n            } else {\n                document.addEventListener("DOMContentLoaded", function() {\n                    LoadHandler_OnDomReady();\n                });\n            }\n        ')

    def _OnDomReady(self):
        if False:
            while True:
                i = 10
        print('DOM is ready!')
        self.browser.ExecuteFunction('alert', 'Message from Python: DOM is ready!')
if __name__ == '__main__':
    main()