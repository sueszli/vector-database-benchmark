from lightning.app import LightningFlow, LightningApp
from lightning.app.testing import EmptyFlow, EmptyWork

class FlowB(LightningFlow):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.flow_d = EmptyFlow()
        self.work_b = EmptyWork()

    def run(self):
        if False:
            print('Hello World!')
        ...

class FlowA(LightningFlow):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.flow_b = FlowB()
        self.flow_c = EmptyFlow()
        self.work_a = EmptyWork()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        ...
app = LightningApp(FlowA())