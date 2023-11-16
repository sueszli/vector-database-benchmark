from lightning import LightningApp, LightningFlow

class Flow(LightningFlow):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.names = []

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.names)

    def add_name(self, name: str):
        if False:
            print('Hello World!')
        'Add a name.'
        print(f'Received name: {name}')
        self.names.append(name)

    def configure_commands(self):
        if False:
            for i in range(10):
                print('nop')
        commands = [{'add': self.add_name}]
        return commands
app = LightningApp(Flow())