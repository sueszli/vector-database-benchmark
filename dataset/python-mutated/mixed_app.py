from .server_app import ServerApp
from .gui_app import GuiApp

class MixedApp(ServerApp, GuiApp):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def initialize(self):
        if False:
            return 10
        super().initialize()

    def run(self):
        if False:
            print('Hello World!')
        super().run()