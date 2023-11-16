from cefpython3 import cefpython as cef
import sys
html = '\n<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <title>Title</title>\n</head>\n<body>\n<script>\n  window.onload = function() {\n   fetch(\'http://127.0.0.1:8000\', {\n       method: \'POST\',\n       headers: {\'Content-Type\': \'application/x-www-form-urlencoded\'},\n       body: \'key=\' + encodeURI(\'🍣 asd\'),\n   }).then().catch();\n  }\n</script>\n</body>\n</html>\n'

class RequestHandler:

    def GetResourceHandler(self, browser, frame, request):
        if False:
            while True:
                i = 10
        print(request.GetPostData())
        return None

def main():
    if False:
        print('Hello World!')
    sys.excepthook = cef.ExceptHook
    cef.Initialize()
    browser = cef.CreateBrowserSync(url=cef.GetDataUrl(html))
    browser.SetClientHandler(RequestHandler())
    cef.MessageLoop()
    del browser
    cef.Shutdown()
if __name__ == '__main__':
    main()