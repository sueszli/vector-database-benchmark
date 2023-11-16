from lightning.app import LightningFlow, LightningApp
from lightning.app.api import Post

class Flow(LightningFlow):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.names = []

    def run(self):
        if False:
            return 10
        print(self.names)

    def handle_post(self, name: str):
        if False:
            print('Hello World!')
        self.names.append(name)
        return f'The name {name} was registered'

    def configure_api(self):
        if False:
            for i in range(10):
                print('nop')
        return [Post(route='/name', method=self.handle_post)]
app = LightningApp(Flow())