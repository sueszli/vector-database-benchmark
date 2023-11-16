"""Create a window that stays on top of other windows."""
import time
import webview

def deactivate(window):
    if False:
        print('Hello World!')
    time.sleep(20)
    window.on_top = False
    window.load_html('<h1>This window is no longer on top of other windows</h1>')
if __name__ == '__main__':
    window = webview.create_window('Topmost window', html='<h1>This window is on top of other windows</h1>', on_top=True)
    webview.start(deactivate, window)