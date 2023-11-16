from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class TrainComponent(LightningWork):

    def run(self, message):
        if False:
            while True:
                i = 10
        for i in range(100000000000):
            print(message, i)

class AnalyzeComponent(LightningWork):

    def run(self, message):
        if False:
            while True:
                i = 10
        for i in range(100000000000):
            print(message, i)

class LitWorkflow(LightningFlow):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.train = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.baseline_1 = TrainComponent(cloud_compute=CloudCompute('cpu'))
        self.analyze = AnalyzeComponent(cloud_compute=CloudCompute('cpu'))

    def run(self):
        if False:
            i = 10
            return i + 15
        self.train.run('machine A counting')
        self.baseline_1.run('machine C counting')
        self.analyze.run('machine B counting')
app = LightningApp(LitWorkflow())