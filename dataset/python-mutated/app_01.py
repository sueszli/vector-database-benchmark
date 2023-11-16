from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.utilities.app_helpers import pretty_state

class Work(LightningWork):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(cache_calls=False)
        self.counter = 0

    def run(self):
        if False:
            return 10
        self.counter += 1

class Flow(LightningFlow):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.w = Work()

    def run(self):
        if False:
            i = 10
            return i + 15
        if self.w.has_started:
            print(f'State: {pretty_state(self.state)} \n')
        self.w.run()
app = LightningApp(Flow())