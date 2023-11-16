"""
Programmatically destroy created window after five seconds.
"""
import time
import webview

def destroy(window):
    if False:
        return 10
    time.sleep(5)
    print('Destroying window..')
    window.destroy()
    print('Destroyed!')
if __name__ == '__main__':
    window = webview.create_window('Destroy Window Example', 'https://pywebview.flowrl.com/hello')
    webview.start(destroy, window)
    print('Window is destroyed')