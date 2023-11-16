from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class TrainComponent(LightningWork):

    def run(self, message):
        if False:
            print('Hello World!')
        for i in range(100000000000):
            print(message, i)

class AnalyzeComponent(LightningWork):

    def run(self, message):
        if False:
            for i in range(10):
                print('nop')
        for i in range(100000000000):
            print(message, i)

class LitWorkflow(LightningFlow):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'), parallel=True)
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('cpu'))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.train.run('machine A counting')
        self.analyze.run('machine B counting')
app = LightningApp(LitWorkflow())