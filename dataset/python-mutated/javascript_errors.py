"""
Two ways for intercepting Javascript errors:
1. window.onerror event in Javascript
2. DisplayHandler.OnConsoleMessage in Python
"""
from cefpython3 import cefpython as cef
g_htmlcode = '\n<!doctype html>\n<html>\n<head>\n    <style>\n    body, html {\n        font-family: Arial;\n        font-size: 11pt;\n    }\n    </style>\n    <script>\n    function print(msg) {\n        document.getElementById("console").innerHTML += msg+"<br>";\n    }\n    window.onerror = function(message, source, lineno, colno, error) {\n        print("[JS:window.onerror] "+error+" (line "+lineno+")");\n        // Return false so that default event handler is fired and\n        // OnConsoleMessage can also intercept this error.\n        return false;\n    };\n    window.onload = function() {\n        forceError();\n    };\n    </script>\n</head>\n<body>\n    <h1>Javascript Errors</h1>\n    <div id=console></div>\n</body>\n</html>\n'

def main():
    if False:
        i = 10
        return i + 15
    cef.Initialize()
    browser = cef.CreateBrowserSync(url=cef.GetDataUrl(g_htmlcode), window_title='Javascript Errors')
    browser.SetClientHandler(DisplayHandler())
    cef.MessageLoop()
    cef.Shutdown()

class DisplayHandler(object):

    def OnConsoleMessage(self, browser, message, line, **_):
        if False:
            for i in range(10):
                print('nop')
        if 'error' in message.lower() or 'uncaught' in message.lower():
            logmsg = '[Py:OnConsoleMessage] {message} (line {line})'.format(message=message, line=line)
            print(logmsg)
            browser.ExecuteFunction('print', logmsg)
if __name__ == '__main__':
    main()