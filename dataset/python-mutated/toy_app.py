from lightning.app import LightningWork, LightningFlow, LightningApp

class TrainComponent(LightningWork):

    def run(self, x):
        if False:
            print('Hello World!')
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):

    def run(self, x):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        self.train.run('CPU machine 1')
        self.analyze.run('CPU machine 2')
app = LightningApp(WorkflowOrchestrator())