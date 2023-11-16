from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class TrainComponent(LightningWork):

    def run(self, x):
        if False:
            print('Hello World!')
        print(f'train a model on {x}')

class AnalyzeComponent(LightningWork):

    def run(self, x):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        self.train.run('GPU machine 1')
        self.train.stop()
        if self.train.status.STOPPED:
            self.analyze.run('CPU machine 2')
app = LightningApp(WorkflowOrchestrator())