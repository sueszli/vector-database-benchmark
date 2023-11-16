from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class TrainComponent(LightningWork):

    def run(self, x):
        if False:
            return 10
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):

    def run(self, x):
        if False:
            return 10
        print(f'analyze model on {x}')

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('gpu'))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.train.run('GPU machine 1')
        if self.schedule('hourly'):
            self.analyze.run('CPU machine 2')
app = LightningApp(WorkflowOrchestrator())