import os
from lightning.app import CloudCompute, LightningApp, LightningWork

class MyWork(LightningWork):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(cloud_compute=CloudCompute(name=os.environ.get('COMPUTE_NAME', 'default')))

    def run(self):
        if False:
            print('Hello World!')
        pass
app = LightningApp(MyWork())