"""Create multiple windows."""
import webview

def third_window():
    if False:
        print('Hello World!')
    third_window = webview.create_window('Window #3', html='<h1>Third Window</h1>')
if __name__ == '__main__':
    master_window = webview.create_window('Window #1', html='<h1>First window</h1>')
    second_window = webview.create_window('Window #2', html='<h1>Second window</h1>')
    webview.start(third_window)