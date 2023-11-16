from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.runners import MultiProcessRuntime

class TrainComponent(LightningWork):

    def run(self, x):
        if False:
            for i in range(10):
                print('nop')
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):

    def run(self, x):
        if False:
            for i in range(10):
                print('nop')
        print(f'analyze model on {x}')

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.train = TrainComponent()
        self.analyze = AnalyzeComponent()

    def run(self):
        if False:
            print('Hello World!')
        self.train.run('GPU machine 1')
        self.analyze.run('CPU machine 2')
app = LightningApp(WorkflowOrchestrator())
MultiProcessRuntime(app).dispatch()