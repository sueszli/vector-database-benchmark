from time import sleep
from lightning.app import LightningWork, LightningFlow, LightningApp

class HourLongWork(LightningWork):

    def __init__(self, parallel: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__(parallel=parallel)
        self.progress = 0.0

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.progress = 0.0
        for _ in range(3600):
            self.progress += 1.0 / 3600
            sleep(1)

class RootFlow(LightningFlow):

    def __init__(self, child_work: LightningWork):
        if False:
            return 10
        super().__init__()
        self.child_work = child_work

    def run(self):
        if False:
            print('Hello World!')
        print(round(self.child_work.progress, 4))
        self.child_work.run()
        if self.child_work.counter == 1.0:
            print('1 hour later!')
app = LightningApp(RootFlow(HourLongWork()))