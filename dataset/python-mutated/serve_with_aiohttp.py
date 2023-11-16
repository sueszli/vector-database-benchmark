"""
Example of serving a Flexx app using a regular web server. In this case aiohttp.

Flexx' own server does quite a lot of things for each connection, which
makes it less suited for long running and/or heavy duty server
processes. Firstly, we don't expect the Flexx server to scale well to
say thousands of connections (tens to a few hundred at a time should
work fine though). Secondly, the amount of work and complexity of each
connection may make the server less stable and potentially vulnerable.

Part of these concerns can be alleviated by running the Flexx server
in an auto-restarting Docker container (as we do with our demo server).

Nevertheless, we want to offer a simple path to build reliable and
performant websites using Flexx. The way that this works is that one
builds the client-side of the app in Flexx, which is then "dumped" (say
exported in-memory) to its bare html/js/css assets, which can be served
by any kind of web server.

"""
import mimetypes
from aiohttp import web
from flexx import flx
from flexxamples.howtos.editor_cm import CodeEditor

class MyApp(flx.Widget):

    def init(self):
        if False:
            while True:
                i = 10
        with flx.HBox():
            CodeEditor(flex=1)
            flx.Widget(flex=1)
app = flx.App(MyApp)
assets = app.dump('index.html', link=2)

def handler(request):
    if False:
        while True:
            i = 10
    path = request.path.lstrip('/') or 'index.html'
    print(request.method, path)
    asset = assets.get(path, None)
    if asset is None:
        return web.Response(status=404, text='not found')
    ct = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    return web.Response(body=asset, content_type=ct)
if __name__ == '__main__':
    app = web.Application()
    app.router.add_get('/', handler)
    app.router.add_get('/{tail:flexx/.*}', handler)
    web.run_app(app, host='0.0.0.0', port=8080)