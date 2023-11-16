"""Exposing Python functions to the Javascript domain."""
import webview

def lol():
    if False:
        print('Hello World!')
    print('LOL')

def wtf():
    if False:
        while True:
            i = 10
    print('WTF')

def echo(arg1, arg2, arg3):
    if False:
        print('Hello World!')
    print(arg1)
    print(arg2)
    print(arg3)

def expose(window):
    if False:
        for i in range(10):
            print('nop')
    window.expose(echo)
    window.evaluate_js('pywebview.api.lol()')
    window.evaluate_js('pywebview.api.wtf()')
    window.evaluate_js('pywebview.api.echo(1, 2, 3)')
if __name__ == '__main__':
    window = webview.create_window('JS Expose Example', html='<html><head></head><body><h1>JS Expost</body></html>')
    window.expose(lol, wtf)
    webview.start(expose, window, debug=True)