from models import NamePostConfig
from lightning.app import LightningFlow, LightningApp
from lightning.app.api import Post

class Flow(LightningFlow):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.names = []

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.names)

    def handle_post(self, config: NamePostConfig):
        if False:
            print('Hello World!')
        self.names.append(config.name)
        return f'The name {config} was registered'

    def configure_api(self):
        if False:
            print('Hello World!')
        return [Post(route='/name', method=self.handle_post)]
app = LightningApp(Flow())