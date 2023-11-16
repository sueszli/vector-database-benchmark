import subprocess
from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute

class ExternalModelServer(LightningWork):

    def run(self, x):
        if False:
            for i in range(10):
                print('nop')
        process = subprocess.Popen('g++ model_server.cpp -o model_server')
        process.wait()
        process = subprocess.Popen('./model_server')
        process.wait()

class LocustLoadTester(LightningWork):

    def run(self, x):
        if False:
            print('Hello World!')
        cmd = f'locust --master-host {self.host} --master-port {self.port}'
        process = subprocess.Popen(cmd)
        process.wait()

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.serve = ExternalModelServer(cloud_compute=CloudCompute('cpu'), parallel=True)
        self.load_test = LocustLoadTester(cloud_compute=CloudCompute('cpu'))

    def run(self):
        if False:
            print('Hello World!')
        self.serve.run()
        if self.serve.state.RUNNING:
            self.load_test.run()
app = LightningApp(WorkflowOrchestrator())