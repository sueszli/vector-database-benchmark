"""
Example of serving a Flexx app using a regular web server. In this case flask.
See serve_with_aiohttp.py for a slightly more advanced example.
"""
from flask import Flask
from flexx import flx
from flexxamples.howtos.editor_cm import CodeEditor

class MyApp(flx.Widget):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        with flx.HBox():
            CodeEditor(flex=1)
            flx.Widget(flex=1)
app = flx.App(MyApp)
assets = app.dump('index.html', link=0)
app = Flask(__name__)

@app.route('/')
def handler():
    if False:
        while True:
            i = 10
    return assets['index.html'].decode()
if __name__ == '__main__':
    app.run(host='localhost', port=8080)