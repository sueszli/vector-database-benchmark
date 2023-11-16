"""Run Javascript code from Python."""
import webview

def evaluate_js(window):
    if False:
        while True:
            i = 10
    result = window.evaluate_js("\n        var h1 = document.createElement('h1')\n        var text = document.createTextNode('Hello pywebview')\n        h1.appendChild(text)\n        document.body.appendChild(h1)\n\n        document.body.style.backgroundColor = '#212121'\n        document.body.style.color = '#f2f2f2'\n\n        // Return user agent\n        'User agent:\\n' + navigator.userAgent;\n        ")
    print(result)
if __name__ == '__main__':
    window = webview.create_window('Run custom JavaScript', html='<html><body></body></html>')
    webview.start(evaluate_js, window)