import webview
'\nLoading custom CSS in a webview window\n'

def load_css(window):
    if False:
        while True:
            i = 10
    window.load_css('body { background: red !important; }')
if __name__ == '__main__':
    window = webview.create_window('Load CSS Example', 'https://pywebview.flowrl.com/hello')
    webview.start(load_css, window)