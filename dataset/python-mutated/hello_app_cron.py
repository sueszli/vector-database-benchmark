from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class TrainComponent(LightningWork):

    def run(self, x):
        if False:
            while True:
                i = 10
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):

    def run(self, x):
        if False:
            i = 10
            return i + 15
        print(f'analyze model on {x}')

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('gpu'))

    def run(self):
        if False:
            return 10
        self.train.run('GPU machine 1')
        if self.schedule('5 4 * * *'):
            self.analyze.run('CPU machine 2')
app = LightningApp(WorkflowOrchestrator())