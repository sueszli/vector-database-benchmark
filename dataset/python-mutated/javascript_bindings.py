"""
Communicate between Python and Javascript asynchronously using
inter-process messaging with the use of Javascript Bindings.
"""
from cefpython3 import cefpython as cef
g_htmlcode = '\n<!doctype html>\n<html>\n<head>\n    <style>\n    body, html {\n        font-family: Arial;\n        font-size: 11pt;\n    }\n    </style>\n    <script>\n    function print(msg) {\n        document.getElementById("console").innerHTML += msg+"<br>";\n    }\n    function js_function(value) {\n        print("Value sent from Python: <b>"+value+"</b>");\n        py_function("I am a Javascript string #1", js_callback);\n    }\n    function js_callback(value, py_callback) {\n        print("Value sent from Python: <b>"+value+"</b>");\n        py_callback("I am a Javascript string #2");\n    }\n    </script>\n</head>\n<body>\n    <h1>Javascript Bindings</h1>\n    <div id=console></div>\n</body>\n</html>\n'

def main():
    if False:
        return 10
    cef.Initialize()
    browser = cef.CreateBrowserSync(url=cef.GetDataUrl(g_htmlcode), window_title='Javascript Bindings')
    browser.SetClientHandler(LoadHandler())
    bindings = cef.JavascriptBindings()
    bindings.SetFunction('py_function', py_function)
    bindings.SetFunction('py_callback', py_callback)
    browser.SetJavascriptBindings(bindings)
    cef.MessageLoop()
    del browser
    cef.Shutdown()

def py_function(value, js_callback):
    if False:
        i = 10
        return i + 15
    print('Value sent from Javascript: ' + value)
    js_callback.Call('I am a Python string #2', py_callback)

def py_callback(value):
    if False:
        return 10
    print('Value sent from Javascript: ' + value)

class LoadHandler(object):

    def OnLoadEnd(self, browser, **_):
        if False:
            print('Hello World!')
        browser.ExecuteFunction('js_function', 'I am a Python string #1')
if __name__ == '__main__':
    main()