from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.pdb import set_trace

class Component(LightningWork):

    def run(self, x):
        if False:
            for i in range(10):
                print('nop')
        print(x)
        set_trace()

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.component = Component()

    def run(self):
        if False:
            print('Hello World!')
        self.component.run('i love Lightning')
app = LightningApp(WorkflowOrchestrator())