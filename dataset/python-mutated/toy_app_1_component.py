from lightning.app import LightningWork, LightningFlow, LightningApp

class Component(LightningWork):

    def run(self, x):
        if False:
            for i in range(10):
                print('nop')
        print(x)

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.component = Component()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.component.run('i love Lightning')
app = LightningApp(WorkflowOrchestrator())