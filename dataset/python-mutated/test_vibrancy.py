import webview

def load_css(window):
    if False:
        return 10
    window.load_css('body { background: transparent !important; }')
if __name__ == '__main__':
    window = webview.create_window('set vibrancy example', 'https://pywebview.flowrl.com/hello', transparent=True, vibrancy=True)
    webview.start(load_css, window)