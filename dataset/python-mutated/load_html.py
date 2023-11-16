from time import sleep
import webview
'\nLoading new HTML after the window is created\n'

def load_html(window):
    if False:
        while True:
            i = 10
    sleep(5)
    window.load_html('<h1>This is dynamically loaded HTML</h1>')
if __name__ == '__main__':
    window = webview.create_window('Load HTML Example', html='<h1>This is initial HTML</h1>')
    webview.start(load_html, window)