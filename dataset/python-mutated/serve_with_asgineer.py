"""
Example of serving a Flexx app using a regular web server. In this case Asgineer.
https://github.com/almarklein/asgineer
"""
import asgineer
from flexx import flx
from flexxamples.howtos.editor_cm import CodeEditor

class MyApp(flx.Widget):

    def init(self):
        if False:
            return 10
        with flx.HBox():
            CodeEditor(flex=1)
            flx.Widget(flex=1)
app = flx.App(MyApp)
assets = app.dump('index.html', link=0)
asset_handler = asgineer.utils.make_asset_handler(assets)

@asgineer.to_asgi
async def main_handler(request):
    return await asset_handler(request)
if __name__ == '__main__':
    asgineer.run(main_handler, 'uvicorn', 'localhost:8080')