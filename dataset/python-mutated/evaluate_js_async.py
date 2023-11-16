import webview
'\nRun asynchronous Javascript code and invoke a callback.\n'

def callback(result):
    if False:
        for i in range(10):
            print('nop')
    print(result)

def evaluate_js_async(window):
    if False:
        while True:
            i = 10
    window.evaluate_js("\n        new Promise((resolve, reject) => {\n            setTimeout(() => {\n                resolve('Whaddup!');\n            }, 300);\n        });\n        ", callback)
if __name__ == '__main__':
    window = webview.create_window('Run async Javascript', html='<html><body></body></html>')
    webview.start(evaluate_js_async, window)