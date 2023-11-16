"""
Change URL ten seconds after the first URL is loaded.
"""
import time
import webview

def change_url(window):
    if False:
        print('Hello World!')
    time.sleep(10)
    window.load_url('https://pywebview.flowrl.com/hello')
if __name__ == '__main__':
    window = webview.create_window('URL Change Example', 'http://www.google.com')
    webview.start(change_url, window)