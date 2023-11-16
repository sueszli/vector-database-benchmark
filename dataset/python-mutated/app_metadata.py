from lightning.app.core.app import LightningApp
from lightning.app.core.flow import LightningFlow
from lightning.app.core.work import LightningWork
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.utilities.packaging.cloud_compute import CloudCompute

class WorkA(LightningWork):

    def __init__(self):
        if False:
            print('Hello World!')
        'WorkA.'
        super().__init__()

    def run(self):
        if False:
            print('Hello World!')
        pass

class WorkB(LightningWork):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'WorkB.'
        super().__init__(cloud_compute=CloudCompute('gpu'))

    def run(self):
        if False:
            print('Hello World!')
        pass

class FlowA(LightningFlow):

    def __init__(self):
        if False:
            while True:
                i = 10
        'FlowA Component.'
        super().__init__()
        self.work_a = WorkA()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class FlowB(LightningFlow):

    def __init__(self):
        if False:
            return 10
        'FlowB.'
        super().__init__()
        self.work_b = WorkB()

    def run(self):
        if False:
            i = 10
            return i + 15
        pass

    def configure_layout(self):
        if False:
            i = 10
            return i + 15
        return StaticWebFrontend(serve_dir='.')

class RootFlow(LightningFlow):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'RootFlow.'
        super().__init__()
        self.flow_a_1 = FlowA()
        self.flow_a_2 = FlowA()
        self.flow_b = FlowB()

    def run(self):
        if False:
            print('Hello World!')
        self.stop()
app = LightningApp(RootFlow())